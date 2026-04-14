import cv2
import numpy as np
from typing import Union, Iterable, Tuple, List
from scipy.spatial import distance


# Fonction d'échantillonnage resserrée sur les points réels
def get_bezier_line(
    group_pts, ref_start, ref_end, is_upper=True, other_side_height=None
):
    group_pts = np.array(group_pts)

    # Identification des extrémités réelles
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

    # CAS A : Déséquilibre (Seulement 2 points, ex: une ligne droite)
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

    # CAS B : Points multiples (ex: 5 points)
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

    # Obtenir l'orientation via le rectangle minimal
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    if len(pts) <= 3:
        pts = box

    # Identifier l'axe long
    dist01 = np.linalg.norm(box[0] - box[1])
    dist12 = np.linalg.norm(box[1] - box[2])
    if dist01 < dist12:
        box = np.roll(box, 1, axis=0)

    # Projection pour séparer Haut et Bas par rapport à la ligne médiane
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
        lower_points, p3_rect, p2_rect, is_upper=False, other_side_height=max_h
    )

    return np.vstack([line1, line2])
