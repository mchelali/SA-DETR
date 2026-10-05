import cv2
import numpy as np
from typing import Union, Iterable, Tuple, List
from scipy.spatial import distance
from scipy.optimize import minimize

try:
    import torch

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False

# ============================================================================
#  Tangent Estimation (NumPy + Torch)
# ============================================================================


def compute_tangents_np(
    anchors: np.ndarray,
    alpha: float | None = 0.5,
    closed: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """
    Vectorized tangent estimation for a polygon (NumPy).

    anchors : (N, 2)
    """
    pts = np.asarray(anchors, dtype=np.float32)
    N = len(pts)

    if closed:
        prev_pts = pts[[i - 1 for i in range(N)]]
        next_pts = pts[(np.arange(N) + 1) % N]
    else:
        prev_pts = np.vstack([pts[0], pts[:-1]])
        next_pts = np.vstack([pts[1:], pts[-1]])

    direction = next_pts - prev_pts
    eps = 1e-8

    if alpha is None:
        # adaptive tangents
        l_prev = np.linalg.norm(pts - prev_pts, axis=1)
        l_next = np.linalg.norm(next_pts - pts, axis=1)
        mean_len = 0.5 * (l_prev + l_next)
        norm = np.linalg.norm(direction, axis=1) + eps
        tang = direction * (mean_len[:, None] / norm[:, None])
    else:
        tang = alpha * direction

    if normalize:
        norm = np.linalg.norm(tang, axis=1, keepdims=True) + eps
        tang /= norm

    return tang


if _TORCH_AVAILABLE:

    def compute_tangents_torch(
        anchors: torch.Tensor,
        alpha: float | None = 0.5,
        closed: bool = True,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Vectorized differentiable tangent estimation.
        anchors: (N,2) or (B,N,2)
        """
        if anchors.ndim == 2:
            anchors = anchors.unsqueeze(0)

        B, N, _ = anchors.shape

        idx_prev = torch.arange(N, device=anchors.device) - 1
        idx_next = torch.arange(N, device=anchors.device) + 1
        if closed:
            idx_prev %= N
            idx_next %= N
        else:
            idx_prev = torch.clamp(idx_prev, 0, N - 1)
            idx_next = torch.clamp(idx_next, 0, N - 1)

        prev_pts = anchors[:, idx_prev]
        next_pts = anchors[:, idx_next]

        direction = next_pts - prev_pts
        eps = 1e-8

        if alpha is None:
            l_prev = (anchors - prev_pts).norm(dim=-1)
            l_next = (next_pts - anchors).norm(dim=-1)
            mean_len = 0.5 * (l_prev + l_next)
            norm = direction.norm(dim=-1) + eps
            tang = direction * (mean_len / norm).unsqueeze(-1)
        else:
            tang = alpha * direction

        if normalize:
            tang = tang / (tang.norm(dim=-1, keepdim=True) + eps)

        return tang.squeeze(0)


# ============================================================================
#  Hermite Basis
# ============================================================================


def hermite_basis_np(u: np.ndarray):
    u = np.asarray(u, dtype=np.float32)
    return np.stack(
        [
            2 * u**3 - 3 * u**2 + 1,
            u**3 - 2 * u**2 + u,
            -2 * u**3 + 3 * u**2,
            u**3 - u**2,
        ],
        axis=0,
    )


if _TORCH_AVAILABLE:

    def hermite_basis_torch(u: torch.Tensor):
        return torch.stack(
            [
                2 * u**3 - 3 * u**2 + 1,
                u**3 - 2 * u**2 + u,
                -2 * u**3 + 3 * u**2,
                u**3 - u**2,
            ],
            dim=0,
        )


# ============================================================================
#  Hermite segment sampling
# ============================================================================


def hermite_segment_np(P0, T0, P1, T1, n=60):
    """Cubic Hermite sampling (NumPy)."""
    u = np.linspace(0, 1, n, dtype=np.float32)
    h00, h10, h01, h11 = hermite_basis_np(u)

    x = h00 * P0[0] + h10 * T0[0] + h01 * P1[0] + h11 * T1[0]
    y = h00 * P0[1] + h10 * T0[1] + h01 * P1[1] + h11 * T1[1]

    return np.stack([x, y], axis=1)


if _TORCH_AVAILABLE:

    def hermite_segment_torch(P0, T0, P1, T1, u):
        """
        P0,T0,P1,T1 : (...,2)
        u : (K,) or (...,K)
        """
        h00, h10, h01, h11 = hermite_basis_torch(u)

        # (broadcast on leading dims)
        def mix(h, p):
            return h.unsqueeze(-1) * p

        x = (
            mix(h00, P0[..., 0])
            + mix(h10, T0[..., 0])
            + mix(h01, P1[..., 0])
            + mix(h11, T1[..., 0])
        )
        y = (
            mix(h00, P0[..., 1])
            + mix(h10, T0[..., 1])
            + mix(h01, P1[..., 1])
            + mix(h11, T1[..., 1])
        )

        return torch.stack([x, y], dim=-1)


# ============================================================================
#  Build the closed spline (NumPy)
# ============================================================================


def build_closed_spline_np(anchors, tangents=None, n_per_seg=60, alpha=0.5):
    pts = np.asarray(anchors, dtype=np.float32)
    if tangents is None:
        tangents = compute_tangents_np(pts, alpha=alpha)

    curves = [
        hermite_segment_np(
            pts[i],
            tangents[i],
            pts[(i + 1) % len(pts)],
            tangents[(i + 1) % len(pts)],
            n=n_per_seg,
        )
        for i in range(len(pts))
    ]
    return np.vstack(curves)


# ============================================================================
#  Arc-length resampling
# ============================================================================


def resample_arc_length_np(curve, samples=25):
    d = np.sqrt(((curve[1:] - curve[:-1]) ** 2).sum(1))
    arc = np.concatenate([[0], np.cumsum(d)])
    arc /= arc[-1]

    target = np.linspace(0, 1, samples, dtype=np.float32)
    idx = np.searchsorted(arc, target)
    idx = np.clip(idx, 1, len(curve) - 1)

    t = (target - arc[idx - 1]) / (arc[idx] - arc[idx - 1] + 1e-8)
    return curve[idx - 1] * (1 - t[:, None]) + curve[idx] * t[:, None]


# ============================================================================
#  High-level API
# ============================================================================


def get_extreme_points_and_orient(pts):
    """
    Trouve les deux points les plus éloignés pour définir l'axe horizontal du texte.
    """
    # Calculer la matrice de distance entre tous les points
    dist_matrix = distance.cdist(pts, pts, "euclidean")
    # Trouver les indices des deux points les plus distants
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

    p1, p2 = pts[i], pts[j]

    # S'assurer que p1 est à gauche de p2 pour le sens de lecture
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    return p1, p2


def polygon_to_spline(
    polygon: np.ndarray,
    samples: int = 25,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Full pipeline: polygon → Hermite spline → arc-length sampling.
    """
    tang = compute_tangents_np(polygon, alpha=alpha)
    dense = build_closed_spline_np(polygon, tangents=tang, n_per_seg=80)
    return resample_arc_length_np(dense, samples=samples)


def sort_polygon_clockwise(coords):
    """
    Prend une liste de coords [x1,y1,...,xn,yn]
    et renvoie les points triés clockwise autour du centroïde.
    """
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)

    # Centroïde
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)

    # Tri par angle (clockwise)
    order = np.argsort(angles)
    pts_sorted = pts[order]

    return pts_sorted  # .reshape(-1).tolist()


def polygon_to_quad_TL_TR_BR_BL(coords):
    pts = np.array(sort_polygon_clockwise(coords)).reshape(-1, 2)

    # ordonner par y, puis par x
    sorted_by_y = pts[np.argsort(pts[:, 1])]

    top = sorted_by_y[:2]
    bottom = sorted_by_y[-2:]

    TL, TR = top[np.argsort(top[:, 0])]
    BL, BR = bottom[np.argsort(bottom[:, 0])]

    # ordre TL, TR, BR, BL
    quad = np.vstack([TL, TR, BR, BL])
    return quad  # .reshape(-1).tolist()


def interp(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    return (1 - t) * p0 + t * p1


def quad_to_bezier(
    quad: np.ndarray, t1: float = 1 / 3, t2: float = 2 / 3
) -> List[float]:
    """Convert quadrilateral to 8-point Bezier representation."""
    quad = polygon_to_quad_TL_TR_BR_BL(quad)
    TL, TR, BR, BL = quad

    TC1 = interp(TL, TR, t1)
    TC2 = interp(TL, TR, t2)
    BC1 = interp(BL, BR, t1)
    BC2 = interp(BL, BR, t2)

    bezier = np.vstack([TL, TC1, TC2, TR, BL, BC1, BC2, BR])
    return bezier  # .tolist()


# 3. Fonction d'échantillonnage resserrée sur les points réels
def get_bezier_line(
    group_pts, ref_start, ref_end, is_upper=True, other_side_height=None
):
    group_pts = np.array(group_pts)

    # 1. Identification des extrémités réelles
    if len(group_pts) >= 2:
        dists_start = np.linalg.norm(group_pts - ref_start, axis=1)
        dists_end = np.linalg.norm(group_pts - ref_end, axis=1)
        actual_start = group_pts[np.argmin(dists_start)]
        actual_end = group_pts[np.argmin(dists_end)]
    else:
        # Si vraiment aucun point n'est trouvé dans ce groupe, on prend les refs du rectangle
        actual_start, actual_end = ref_start, ref_end

    # Vecteur directeur et perpendiculaire
    vec = actual_end - actual_start
    dx, dy = vec[0], vec[1]
    perp_vec = np.array([-dy, dx])
    norm = np.linalg.norm(perp_vec)
    if norm > 0:
        perp_vec /= norm
    direction = 1 if is_upper else -1

    # 2. CAS A : Déséquilibre (Seulement 2 points, ex: une ligne droite)
    if len(group_pts) <= 2:
        # On place les points de contrôle aux tiers de la ligne droite
        cp1_raw = actual_start + vec * 1 / 3
        cp2_raw = actual_start + vec * 2 / 3

        # Si on connaît la hauteur de l'autre côté, on peut simuler une légère courbure
        # Sinon, on reste sur la ligne droite
        offset = other_side_height * 0.2 if other_side_height else 0
        cp1 = cp1_raw + perp_vec * offset * direction
        cp2 = cp2_raw + perp_vec * offset * direction

        return [actual_start, cp1, cp2, actual_end]

    # 3. CAS B : Points multiples (ex: 5 points)
    # Tri par projection
    vec_norm_sq = np.linalg.norm(vec) ** 2 if np.linalg.norm(vec) > 0 else 1
    sorted_pts = np.array(
        sorted(group_pts, key=lambda p: np.dot(p - actual_start, vec) / vec_norm_sq)
    )

    # Distance cumulée pour échantillonnage équitable
    segments = np.sqrt(np.sum(np.diff(sorted_pts, axis=0) ** 2, axis=1))
    cumulative_dist = np.insert(np.cumsum(segments), 0, 0)
    total_length = cumulative_dist[-1]

    def find_nearest_to_dist(target_dist):
        idx = np.argmin(np.abs(cumulative_dist - target_dist))
        return sorted_pts[idx]

    edge_cp1 = find_nearest_to_dist(total_length * 1 / 3)
    edge_cp2 = find_nearest_to_dist(total_length * 2 / 3)

    # Calcul de l'éloignement (Push)
    def push_point(p, start, end, dir_val):
        h = abs(np.cross(end - start, p - start)) / np.linalg.norm(end - start)
        return p + perp_vec * (h * 0.35) * dir_val

    cp1 = push_point(edge_cp1, actual_start, actual_end, direction)
    cp2 = push_point(edge_cp2, actual_start, actual_end, direction)

    return [actual_start, cp1, cp2, actual_end]


def robust_polygon_to_bezier(segmentation_pts):
    """
    Version corrigée : utilise les points réels pour éviter l'écartement des extrémités
    et prépare les données pour un échantillonnage à num_pts_target.
    """
    pts = np.array(segmentation_pts).reshape(-1, 2).astype(np.float32)

    # 1. Obtenir l'orientation via le rectangle minimal
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    if len(pts) <= 3:
        pts = box

    # Identifier l'axe long
    dist01 = np.linalg.norm(box[0] - box[1])
    dist12 = np.linalg.norm(box[1] - box[2])
    if dist01 < dist12:
        box = np.roll(box, 1, axis=0)

    # 2. Projection pour séparer Haut et Bas par rapport à la ligne médiane
    p0_rect, p1_rect, p2_rect, p3_rect = box[0], box[1], box[2], box[3]
    mid_l, mid_r = (p0_rect + p3_rect) / 2, (p1_rect + p2_rect) / 2

    upper_points = []
    lower_points = []
    for p in pts:
        if np.cross(mid_r - mid_l, p - mid_l) > 0:
            upper_points.append(p)
        else:
            lower_points.append(p)

    # Calculer une hauteur de référence si un côté est pauvre en points
    def get_avg_height(g_pts, start, end):
        if len(g_pts) < 3:
            return 0
        heights = [
            abs(np.cross(end - start, p - start)) / np.linalg.norm(end - start)
            for p in g_pts
        ]
        return np.mean(heights)

    h_upper = get_avg_height(upper_points, p0_rect, p1_rect)
    h_lower = get_avg_height(lower_points, p3_rect, p2_rect)
    # On utilise la hauteur max des deux côtés pour harmoniser si besoin
    max_h = max(h_upper, h_lower)

    line1 = get_bezier_line(
        upper_points, p0_rect, p1_rect, is_upper=True, other_side_height=max_h
    )
    line2 = get_bezier_line(
        lower_points, p2_rect, p3_rect, is_upper=False, other_side_height=max_h
    )

    return np.vstack([line1, line2])


##################################################################################
def fit_bezier_to_points(pts, p0, p3, curvature_scale=1.0):
    """
    curvature_scale : 0.0 = droite forcée, 1.0 = fit libre
    """
    pts = np.array(pts, dtype=float)
    p0, p3 = np.array(p0, dtype=float), np.array(p3, dtype=float)

    if len(pts) <= 2:
        return [p0, p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3, p3]

    dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0], np.cumsum(dists)])
    total = cum[-1]
    if total < 1e-6:
        return [p0, p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3, p3]
    t_vals = cum / total

    def bezier(t, cp1, cp2):
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * cp1
            + 3 * (1 - t) * t**2 * cp2
            + t**3 * p3
        )

    def loss(params):
        cp1 = np.array(params[:2])
        cp2 = np.array(params[2:])
        residuals = [bezier(t, cp1, cp2) - pts[i] for i, t in enumerate(t_vals)]
        return sum(np.dot(r, r) for r in residuals)

    init = np.concatenate([p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3])
    res = minimize(loss, init, method="L-BFGS-B")
    cp1_free = res.x[:2]
    cp2_free = res.x[2:]

    # Interpolation entre la droite et le fit libre selon curvature_scale
    cp1_straight = p0 + (p3 - p0) / 3
    cp2_straight = p0 + 2 * (p3 - p0) / 3
    cp1 = cp1_straight + curvature_scale * (cp1_free - cp1_straight)
    cp2 = cp2_straight + curvature_scale * (cp2_free - cp2_straight)

    return [p0, cp1, cp2, p3]


def compute_curvature_ratio(pts, p_start, p_end):
    """
    Mesure l'écart moyen des points par rapport à la droite p_start->p_end,
    normalisé par la longueur du segment.
    0.0 = points parfaitement alignés (rectangle)
    1.0+ = forte courbure (ovale)
    """
    if len(pts) < 3:
        return 0.0
    seg = np.array(p_end) - np.array(p_start)
    seg_len = max(np.linalg.norm(seg), 1e-6)
    seg_unit = seg / seg_len

    heights = [abs(np.cross(seg_unit, np.array(p) - np.array(p_start))) for p in pts]
    avg_h = float(np.mean(heights))
    # Normalisation : rapport hauteur/longueur
    # Un demi-cercle parfait donne ~0.5, un rectangle plat ~0.0
    return avg_h / seg_len


def get_extremities_on_axis(pts, axis_start, axis_end):
    """
    Retourne les deux points de `pts` aux extrémités projetées sur l'axe
    (axis_start -> axis_end), c'est-à-dire le point de projection minimale
    et le point de projection maximale.
    """
    vec = axis_end - axis_start
    norm_sq = np.dot(vec, vec)
    if norm_sq < 1e-10:
        return pts[0], pts[-1]
    projections = [np.dot(p - axis_start, vec) / norm_sq for p in pts]
    idx_min = int(np.argmin(projections))
    idx_max = int(np.argmax(projections))
    return pts[idx_min], pts[idx_max]


def get_bezier_line_v2(group_pts, ref_start, ref_end, fallback_height=0.0):
    ref_start = np.array(ref_start, dtype=float)
    ref_end = np.array(ref_end, dtype=float)

    if len(group_pts) == 0:
        cp1 = ref_start + (ref_end - ref_start) / 3
        cp2 = ref_start + 2 * (ref_end - ref_start) / 3
        return [ref_start, cp1, cp2, ref_end]

    group_pts = np.array(group_pts, dtype=float)
    actual_start, actual_end = get_extremities_on_axis(group_pts, ref_start, ref_end)

    if len(group_pts) <= 2:
        vec = actual_end - actual_start
        perp = np.array([-vec[1], vec[0]])
        norm = np.linalg.norm(perp)
        if norm > 1e-6:
            perp /= norm
        offset = fallback_height * 0.15
        cp1 = actual_start + vec / 3 + perp * offset
        cp2 = actual_start + 2 * vec / 3 + perp * offset
        return [actual_start, cp1, cp2, actual_end]

    vec = actual_end - actual_start
    norm_sq = max(np.dot(vec, vec), 1e-10)
    sorted_pts = sorted(
        group_pts, key=lambda p: np.dot(p - actual_start, vec) / norm_sq
    )

    # Mesure de la courbure naturelle des points
    # ratio = compute_curvature_ratio(sorted_pts, actual_start, actual_end)

    # Seuil bas (rectangle) -> scale faible, seuil haut (ovale) -> scale fort
    # ratio ~0.02-0.05 pour rectangle, ~0.3-0.5 pour ovale
    # curvature_scale = float(np.clip((ratio - 0.02) / (0.25 - 0.02), 0.0, 1.0))
    curvature_scale = 1.0

    return fit_bezier_to_points(sorted_pts, actual_start, actual_end, curvature_scale)


def robust_polygon_to_bezier_v2(segmentation_pts):
    """
    Version robuste pour polygones quelconques (ovale, rectangle coupé, 4+ points).
    Retourne np.vstack([line1, line2]) : 4 pts pour chaque courbe de Bézier.
    """
    pts = np.array(segmentation_pts).reshape(-1, 2).astype(np.float32)

    # Rectangle englobant minimal pour orienter
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)

    if len(pts) <= 3:
        pts = box.astype(np.float32)

    # Identifier l'axe long du rectangle
    dist01 = np.linalg.norm(box[0] - box[1])
    dist12 = np.linalg.norm(box[1] - box[2])
    if dist01 < dist12:
        box = np.roll(box, 1, axis=0)

    p0, p1, p2, p3 = box[0], box[1], box[2], box[3]

    # Ligne médiane (axe long)
    mid_l = (p0 + p3) / 2
    mid_r = (p1 + p2) / 2
    mid_vec = mid_r - mid_l

    # Séparation haut/bas via cross-product sur la ligne médiane
    upper_points, lower_points = [], []
    for p in pts:
        cross = np.cross(mid_vec, p - mid_l)
        if cross >= 0:
            upper_points.append(p)
        else:
            lower_points.append(p)

    # Hauteur de référence pour le côté vide (bord d'image)
    def avg_height(gpts, s, e):
        if len(gpts) < 2:
            return 0.0
        seg = np.array(e) - np.array(s)
        seg_norm = max(np.linalg.norm(seg), 1e-6)
        return float(
            np.mean(
                [abs(np.cross(seg / seg_norm, np.array(p) - np.array(s))) for p in gpts]
            )
        )

    h_upper = avg_height(upper_points, p0, p1)
    h_lower = avg_height(lower_points, p2, p3)
    fallback_h = max(h_upper, h_lower)

    line1 = get_bezier_line_v2(upper_points, p0, p1, fallback_height=fallback_h)
    line2 = get_bezier_line_v2(lower_points, p2, p3, fallback_height=fallback_h)

    return np.vstack([line1, line2])


##############################################

from enum import Enum


class ShapeType(Enum):
    RECTANGLE = "rectangle"  # droite des deux côtés
    OVAL = "oval"  # fit libre des deux côtés
    ROUNDED_RECT = "rounded_rect"  # légère courbure symétrique
    PARTIAL = "partial"  # un côté plat, un côté courbe (bord image)


def compute_curvature_ratio(pts, p_start, p_end):
    if len(pts) < 3:
        return 0.0
    seg = np.array(p_end) - np.array(p_start)
    seg_len = max(np.linalg.norm(seg), 1e-6)
    seg_unit = seg / seg_len
    heights = [abs(np.cross(seg_unit, np.array(p) - np.array(p_start))) for p in pts]
    return float(np.mean(heights)) / seg_len


def classify_shape(
    ratio_upper, ratio_lower, n_upper, n_lower, thresh_flat=0.05, thresh_curved=0.20
):
    """
    Classifie la forme selon les ratios de courbure de chaque côté.

    thresh_flat   : en dessous → côté considéré plat (rectangle)
    thresh_curved : au dessus  → côté considéré courbe (ovale)
    Entre les deux : arrondi léger.
    """
    # Un côté sans points = bord image → on le traite comme plat
    if n_upper == 0:
        ratio_upper = 0.0
    if n_lower == 0:
        ratio_lower = 0.0

    upper_flat = ratio_upper < thresh_flat
    upper_curved = ratio_upper > thresh_curved
    lower_flat = ratio_lower < thresh_flat
    lower_curved = ratio_lower > thresh_curved

    if upper_flat and lower_flat:
        return ShapeType.RECTANGLE

    if upper_curved and lower_curved:
        return ShapeType.OVAL

    # Un côté courbe, l'autre plat → partiellement coupé ou bord image
    if (upper_curved and lower_flat) or (upper_flat and lower_curved):
        return ShapeType.PARTIAL

    # Cas restants : légèrement arrondi des deux côtés
    return ShapeType.ROUNDED_RECT


def get_bezier_for_side(
    group_pts,
    ref_start,
    ref_end,
    ratio,
    shape_side,
    thresh_flat=0.05,
    thresh_curved=0.20,
):
    """
    shape_side : 'flat', 'rounded', 'curved'
    Retourne [p0, cp1, cp2, p3].
    """
    ref_start = np.array(ref_start, dtype=float)
    ref_end = np.array(ref_end, dtype=float)

    if len(group_pts) == 0:
        cp1 = ref_start + (ref_end - ref_start) / 3
        cp2 = ref_start + 2 * (ref_end - ref_start) / 3
        return [ref_start, cp1, cp2, ref_end]

    group_pts = np.array(group_pts, dtype=float)
    actual_start, actual_end = get_extremities_on_axis(group_pts, ref_start, ref_end)

    vec = actual_end - actual_start
    norm_sq = max(np.dot(vec, vec), 1e-10)
    sorted_pts = sorted(
        group_pts, key=lambda p: np.dot(p - actual_start, vec) / norm_sq
    )

    if shape_side == "flat":
        # Droite pure entre les extrémités réelles
        cp1 = actual_start + vec / 3
        cp2 = actual_start + 2 * vec / 3
        return [actual_start, cp1, cp2, actual_end]

    if shape_side == "curved":
        # Fit libre
        return fit_bezier_to_points(
            sorted_pts, actual_start, actual_end, curvature_scale=1.0
        )

    # "rounded" : fit libre mais bridé — curvature_scale proportionnel au ratio
    scale = float(
        np.clip((ratio - thresh_flat) / (thresh_curved - thresh_flat), 0.1, 0.6)
    )
    return fit_bezier_to_points(
        sorted_pts, actual_start, actual_end, curvature_scale=scale
    )


def robust_polygon_to_bezier_v3(segmentation_pts, thresh_flat=0.05, thresh_curved=0.20):
    pts = np.array(segmentation_pts).reshape(-1, 2).astype(np.float32)

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    if len(pts) <= 3:
        pts = box.astype(np.float32)

    dist01 = np.linalg.norm(box[0] - box[1])
    dist12 = np.linalg.norm(box[1] - box[2])
    if dist01 < dist12:
        box = np.roll(box, 1, axis=0)

    p0, p1, p2, p3 = box[0], box[1], box[2], box[3]
    mid_l = (p0 + p3) / 2
    mid_r = (p1 + p2) / 2
    mid_vec = mid_r - mid_l

    upper_points, lower_points = [], []
    for p in pts:
        if np.cross(mid_vec, p - mid_l) >= 0:
            upper_points.append(p)
        else:
            lower_points.append(p)

    ratio_upper = compute_curvature_ratio(upper_points, p0, p1)
    ratio_lower = compute_curvature_ratio(lower_points, p2, p3)

    shape = classify_shape(
        ratio_upper,
        ratio_lower,
        n_upper=len(upper_points),
        n_lower=len(lower_points),
        thresh_flat=thresh_flat,
        thresh_curved=thresh_curved,
    )

    # Décision côté par côté
    def side_type(ratio, n_pts):
        if n_pts == 0 or ratio < thresh_flat:
            return "flat"
        if ratio > thresh_curved:
            return "curved"
        return "rounded"

    upper_side = side_type(ratio_upper, len(upper_points))
    lower_side = side_type(ratio_lower, len(lower_points))

    line1 = get_bezier_for_side(
        upper_points, p0, p1, ratio_upper, upper_side, thresh_flat, thresh_curved
    )
    line2 = get_bezier_for_side(
        lower_points, p2, p3, ratio_lower, lower_side, thresh_flat, thresh_curved
    )

    return np.vstack([line1, line2])  # , shape  # shape utile pour debug/log


##################################################################################
# v4 : conversion polygone -> 2 Béziers cubiques, choix des coins par IoU
#
# Corrige les défauts de v3 (cf. utilities/README_add_bezier2coco.md) :
#  - un point posé sur un petit côté ne fait plus basculer un côté en "courbe" :
#    les coins sont choisis sur le polygone simplifié et la coupe haut/bas suit
#    l'ordre des sommets (de coin en coin) au lieu de la ligne médiane ;
#  - l'ajustement se fait par moindres carrés sur le contour densifié (les
#    arêtes, pas seulement les sommets) : la courbe ne déborde plus ;
#  - plusieurs jeux de coins sont essayés, le contour qui se croise est rejeté
#    et celui de meilleure IoU avec le polygone est retenu ;
#  - sortie dans un ordre canonique : haut de gauche à droite (P0 -> P3) puis
#    bas de droite à gauche, dans le repère du grand axe. Le sens « gauche ->
#    droite » de l'axe est fixé par son angle (cf. DIRECTION_CUT_DEG), pas par
#    le sens de lecture du texte, que le polygone seul ne permet pas de connaître.
##################################################################################

from shapely.geometry import Polygon as _ShapelyPolygon


def _signed_area(pts):
    """Aire signée (repère image, y vers le bas) : > 0 pour un parcours horaire à l'écran."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _densify_closed(pts, step):
    """
    Échantillonne le contour fermé tous les `step` pixels environ.

    Les sommets d'origine sont conservés ; renvoie le contour dense et l'indice
    de chaque sommet d'origine dans ce contour.
    """
    dense, vertex_idx = [], []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        vertex_idx.append(len(dense))
        for j in range(k):
            dense.append(a + (b - a) * j / k)
    return np.array(dense, dtype=float), np.array(vertex_idx)


def _cubic_basis(t):
    t = np.asarray(t, dtype=float)[:, None]
    return np.hstack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3])


def _fit_cubic_fixed_ends(chain, n_iter=3):
    """
    Bézier cubique de moindres carrés sur une suite ordonnée de points.

    Les extrémités sont fixées au premier et au dernier point ; P1 et P2 sont
    résolus en forme fermée, puis les paramètres t sont recalés `n_iter` fois
    sur le point de courbe le plus proche.
    """
    p0, p3 = chain[0], chain[-1]
    straight = [p0, p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3, p3]
    if len(chain) < 3:
        return np.array(straight)

    seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    total = seg.sum()
    if total < 1e-6:
        return np.array(straight)
    t = np.concatenate([[0.0], np.cumsum(seg)]) / total  # paramétrage par longueur de corde

    t_fine = np.linspace(0, 1, 400)
    ctrl = np.array(straight)
    for _ in range(n_iter + 1):
        basis = _cubic_basis(t)
        rhs = chain - np.outer(basis[:, 0], p0) - np.outer(basis[:, 3], p3)
        sol, *_ = np.linalg.lstsq(basis[:, 1:3], rhs, rcond=None)
        ctrl = np.vstack([p0, sol, p3])
        # Recalage : chaque point prend le t du point de courbe le plus proche
        curve = _cubic_basis(t_fine) @ ctrl
        d = np.linalg.norm(chain[:, None, :] - curve[None, :, :], axis=2)
        t = t_fine[np.argmin(d, axis=1)]
        t[0], t[-1] = 0.0, 1.0
    return ctrl


def _chain(dense, i_start, i_end):
    """Points du contour dense de i_start à i_end inclus, dans le sens du parcours."""
    if i_end >= i_start:
        return dense[i_start : i_end + 1]
    return np.vstack([dense[i_start:], dense[: i_end + 1]])


def _sample_bezier_boundary(ctrl8, n=25):
    """Contour fermé (2n points) des 2 Béziers, même calcul que adet/data/datasets/text.py."""
    basis = _cubic_basis(np.linspace(0, 1, n))
    return np.vstack([basis @ ctrl8[:4], basis @ ctrl8[4:]])


# Angle (degrés, repère image : 0 = horizontal, 90 = vertical vers le bas) où le
# sens de l'axe bascule. L'axe est orienté pour que son angle soit dans
# [DIRECTION_CUT_DEG - 180, DIRECTION_CUT_DEG) : un tampon horizontal va de
# gauche à droite, un tampon vertical de bas en haut (horizontal tourné d'un
# quart de tour à gauche). 55° est l'orientation la moins fréquente sur Forbin
# (29 tampons sur 4509 à ±5°, contre 221 autour de la verticale).
DIRECTION_CUT_DEG = 55.0


def _frame(box_center, u):
    """Repère (u, v) : u le long de l'axe, orienté selon DIRECTION_CUT_DEG ; v vers le « bas »."""
    phi = np.degrees(np.arctan2(u[1], u[0]))
    if not (DIRECTION_CUT_DEG - 180 <= phi < DIRECTION_CUT_DEG):
        u = -u
    v = np.array([-u[1], u[0]])
    return u, v


def robust_polygon_to_bezier_v4(
    segmentation_pts, simplify_ratio=0.03, try_short_axis_margin=0.01, square_ratio=0.85
):
    """
    Convertit un polygone de segmentation en 8 points de Bézier (2 cubiques).

    Retourne un tableau (8, 2) : [haut P0..P3 (gauche -> droite),
    bas P0..P3 (droite -> gauche)], le bas partant du côté où finit le haut.

    Paramètres :
      simplify_ratio : tolérance de Douglas-Peucker, en fraction du petit côté
          du rectangle minimal. Sert uniquement à choisir les coins : les
          sommets quasi alignés sur un côté ne peuvent pas devenir des coins.
      try_short_axis_margin, square_ratio : pour un tampon presque carré
          (petit côté / grand côté > square_ratio), où le grand axe est
          arbitraire, le petit axe est aussi essayé ; il n'est retenu que s'il
          améliore l'IoU d'au moins try_short_axis_margin. Sinon, le grand axe
          est imposé pour garder une direction cohérente entre tampons.
    """
    pts = np.asarray(segmentation_pts, dtype=float).reshape(-1, 2)
    # Doublons consécutifs
    keep = np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1) > 1e-6
    pts = pts[keep] if keep.sum() >= 3 else pts

    (cx, cy), (w, h), angle = cv2.minAreaRect(pts.astype(np.float32))
    if len(pts) < 3 or min(w, h) < 1e-6:
        pts = cv2.boxPoints(((cx, cy), (max(w, 1.0), max(h, 1.0)), angle)).astype(float)
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts.astype(np.float32))

    # Parcours horaire à l'écran : TL -> TR -> BR -> BL
    if _signed_area(pts) < 0:
        pts = pts[::-1]

    target = _ShapelyPolygon(pts)
    if not target.is_valid:
        target = target.buffer(0)

    # Polygone simplifié : candidats pour les coins
    eps = simplify_ratio * min(w, h)
    simp = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), eps, True)
    simp = simp.reshape(-1, 2).astype(float)
    if len(simp) < 4:
        simp = pts

    step = max(min(w, h), max(w, h)) / 150.0
    dense, vidx = _densify_closed(pts, max(step, 0.5))
    # Indice dense des sommets simplifiés (ce sont des sommets d'origine)
    simp_idx = np.array(
        [vidx[np.argmin(np.linalg.norm(pts - s, axis=1))] for s in simp]
    )

    center = np.array([cx, cy])
    theta = np.deg2rad(angle)
    axis_w = np.array([np.cos(theta), np.sin(theta)])  # axe de longueur w
    axis_h = np.array([-np.sin(theta), np.cos(theta)])  # axe de longueur h
    long_axis, length, width = (axis_w, w, h) if w >= h else (axis_h, h, w)
    short_axis = np.array([-long_axis[1], long_axis[0]])

    def corner_sets(u_axis, L, W):
        u, v = _frame(center, u_axis)
        proj = (dense - center) @ u
        i_left, i_right = int(np.argmin(proj)), int(np.argmax(proj))
        sets = []
        # (0) polygone simplifié à 4 sommets : ces sommets sont les coins. Parmi
        # les 4 rotations, on prend celle dont le côté TL->TR va le plus vers la
        # droite et se trouve le plus en haut (cas des parallélogrammes penchés,
        # où deux coins du rectangle minimal tombent sur le même sommet).
        if len(simp_idx) == 4:
            def top_score(k):
                a, b = dense[simp_idx[k]], dense[simp_idx[(k + 1) % 4]]
                return (b - a) @ u / L - ((a + b) / 2 - center) @ v / max(W, 1e-6)
            k = max(range(4), key=top_score)
            sets.append([int(simp_idx[(k + j) % 4]) for j in range(4)])
        # (a) sommets simplifiés les plus proches des coins du rectangle
        box = [
            center - u * L / 2 - v * W / 2,  # TL
            center + u * L / 2 - v * W / 2,  # TR
            center + u * L / 2 + v * W / 2,  # BR
            center - u * L / 2 + v * W / 2,  # BL
        ]
        sets.append(
            [int(simp_idx[np.argmin(np.linalg.norm(simp - c, axis=1))]) for c in box]
        )
        # (b) extrémités du grand axe (formes ovales : pas de petit côté droit)
        sets.append([i_left, i_right, i_right, i_left])
        # (c) points du contour dense les plus proches des coins (coins arrondis)
        sets.append([int(np.argmin(np.linalg.norm(dense - c, axis=1))) for c in box])
        return sets

    def build(corners):
        tl, tr, br, bl = corners
        n = len(dense)
        # Les coins doivent se suivre dans le sens du parcours : TL, TR, BR, BL.
        # BL peut être confondu avec TL (et BR avec TR) : petit côté de longueur nulle.
        o_tr, o_br = (tr - tl) % n, (br - tl) % n
        o_bl = (bl - tl) % n or n
        if not (0 < o_tr <= o_br < o_bl <= n):
            return None
        top = _fit_cubic_fixed_ends(_chain(dense, tl, tr))
        bottom = _fit_cubic_fixed_ends(_chain(dense, br, bl))
        return np.vstack([top, bottom])

    def score(ctrl8):
        poly = _ShapelyPolygon(_sample_bezier_boundary(ctrl8, 50))
        if not poly.is_valid:  # contour qui se croise : rejeté
            return -1.0
        union = poly.union(target).area
        return poly.intersection(target).area / union if union > 0 else 0.0

    axes = [(long_axis, length, width)]
    if width / max(length, 1e-6) > square_ratio:
        axes.append((short_axis, width, length))

    best, best_iou = None, -2.0
    for axis_idx, (u_axis, L, W) in enumerate(axes):
        margin = 0.0 if axis_idx == 0 else try_short_axis_margin
        for corners in corner_sets(u_axis, L, W):
            ctrl8 = build(corners)
            if ctrl8 is None:
                continue
            iou = score(ctrl8)
            if iou > best_iou + margin:
                best, best_iou = ctrl8, iou

    if best is None:
        # Repli : rectangle minimal, côtés droits
        u, v = _frame(center, long_axis)
        tl = center - u * length / 2 - v * width / 2
        tr = center + u * length / 2 - v * width / 2
        br = center + u * length / 2 + v * width / 2
        bl = center - u * length / 2 + v * width / 2
        best = np.array(
            [tl, tl + (tr - tl) / 3, tl + 2 * (tr - tl) / 3, tr,
             br, br + (bl - br) / 3, br + 2 * (bl - br) / 3, bl]
        )
    return best
