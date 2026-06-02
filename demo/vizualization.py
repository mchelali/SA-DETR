import argparse
import os
import cv2
import json
import numpy as np
import logging
import tqdm

WINDOW_NAME = "Detections"


def setup_logger(name="COCOVisualizer"):
    """Setup a simple logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class SimpleVisualizer:
    """Simple visualizer for drawing polygons on images using OpenCV."""

    def __init__(self, image_rgb):
        """
        Args:
            image_rgb (np.ndarray): Image in RGB format
        """
        self.image_rgb = image_rgb.copy()
        self.output_image = image_rgb.copy()

    def draw_polygon(self, points, color=(0, 1, 0), alpha=0.5, thickness=2):
        """
        Draw a polygon on the image.

        Args:
            points (np.ndarray): Array of polygon points in shape (N, 2)
            color (tuple): RGB color as (R, G, B) with values 0-1
            alpha (float): Transparency (0-1)
            thickness (int): Line thickness in pixels
        """
        if len(points) < 3:
            return

        # Convert color from [0, 1] to [0, 255]
        color_bgr = tuple(int(c * 255) for c in reversed(color))

        # Draw filled polygon with alpha blending
        overlay = self.output_image.copy()
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color_bgr)
        cv2.addWeighted(
            overlay, alpha, self.output_image, 1 - alpha, 0, self.output_image
        )

        # Draw polygon outline
        cv2.polylines(self.output_image, [pts], True, color_bgr, thickness)

    def draw_box(self, bbox, color=(0, 1, 0), alpha=0.5, thickness=2):
        """
        Draw a bounding box with fill and outline.

        Args:
            bbox (list): [x1, y1, x2, y2]
            color (tuple): RGB color
            alpha (float): Transparency
            thickness (int): Line thickness
        """

        x1, y1, x2, y2 = [int(x) for x in bbox]
        color_bgr = tuple(int(c * 255) for c in reversed(color))

        # Draw filled rectangle with alpha blending
        overlay = self.output_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)  # -1 for filled
        cv2.addWeighted(
            overlay, alpha, self.output_image, 1 - alpha, 0, self.output_image
        )

        # Draw rectangle outline
        cv2.rectangle(self.output_image, (x1, y1), (x2, y2), color_bgr, thickness)

    def get_image(self):
        """Get the current image as numpy array (RGB format)."""
        return self.output_image.copy()


class COCOVisualizer:
    """
    Visualization class for COCO format annotations.
    Takes predicted and ground truth JSON files and visualizes detections as polygons only.
    """

    def __init__(self, image_path, instance_mode="segmentation"):
        """
        Args:
            image_path (str): Path to the image file
            instance_mode (str): "bbox" for bounding boxes or "segmentation" for segmentation masks
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB for proper visualization
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.instance_mode = instance_mode
        self.visualizer = SimpleVisualizer(self.image_rgb)

    def draw_from_coco_json(
        self,
        coco_data,
        image_id,
        pred_color=None,
        label_prefix="",
        alpha=0.5,
        score_threshold=None,
    ):
        """
        Draw annotations from COCO format JSON data.

        Args:
            coco_data (dict): COCO format data with 'annotations' and 'images'
            image_id (int): ID of the image to visualize
            pred_color (tuple): Color for predictions as RGB (0-1). If None, random colors used.
            label_prefix (str): Prefix for labels (e.g., "GT", "Pred")
            alpha (float): Transparency for masks
            score_threshold (float): Minimum prediction score to draw. Annotations
                without a score, such as GT annotations, are not filtered.
        """
        # Find annotations for this image
        annotations = [
            ann
            for ann in coco_data.get("annotations", [])
            if ann.get("image_id") == image_id
            and (
                score_threshold is None
                or "score" not in ann
                or ann["score"] >= score_threshold
            )
        ]

        if not annotations:
            return self.visualizer.output_image

        # Define colors (RGB format, 0-1)
        colors = [
            (0, 1, 0),  # green
            (0, 0, 1),  # red
            (1, 0, 0),  # blue
            (1, 1, 0),  # cyan
            (1, 0, 1),  # magenta
            (0, 1, 1),  # yellow
            (0.5, 0, 0.5),  # purple
            (0.5, 0.5, 0),  # olive
        ]

        for idx, annotation in enumerate(annotations):
            color = pred_color if pred_color else colors[idx % len(colors)]

            # Draw segmentation mask if available (polygons)
            if "segmentation" in annotation and annotation["segmentation"]:
                seg = annotation["segmentation"]
                if isinstance(seg, list) and len(seg) > 0:
                    if isinstance(seg[0], list):
                        # Polygon segmentation
                        for poly in seg:
                            if len(poly) > 4:  # At least 2 points
                                points = np.array(poly, dtype=np.int32).reshape(-1, 2)
                                self.visualizer.draw_polygon(
                                    points, color=color, alpha=alpha
                                )
                    elif isinstance(seg, dict) and "counts" in seg:
                        # RLE encoded segmentation - skip
                        pass

            # Also draw bounding box if available (as polygon outline)
            elif "bbox" in annotation:
                bbox = annotation["bbox"]
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                self.visualizer.draw_box([x, y, x + w, y + h], alpha=alpha, color=color)
            else:
                pass

        return self.visualizer.output_image

    def draw_predictions_vs_gt(
        self, pred_json_data, gt_json_data, image_id, alpha=0.5, score_threshold=None
    ):
        """
        Draw both predictions and ground truth on the same image for comparison.

        Args:
            pred_json_data (dict): COCO format predictions
            gt_json_data (dict): COCO format ground truth
            image_id (int): ID of the image to visualize
            alpha (float): Transparency for masks
            score_threshold (float): Minimum prediction score to draw.
        """
        # Draw GT in green
        self.draw_from_coco_json(
            gt_json_data, image_id, pred_color=(0, 1, 0), label_prefix="GT", alpha=alpha
        )

        # Draw predictions in red
        self.draw_from_coco_json(
            pred_json_data,
            image_id,
            pred_color=(1, 0, 0),
            label_prefix="Pred",
            alpha=alpha,
            score_threshold=score_threshold,
        )

        return self.visualizer.output_image

    def save_output(self, output_path):
        """Save visualization to file."""
        image_output = self.visualizer.get_image()
        cv2.imwrite(output_path, cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR))

    def get_image(self):
        """Get the visualization as numpy array (BGR format for OpenCV)."""
        image_output = self.visualizer.get_image()
        return cv2.cvtColor(image_output, cv2.COLOR_RGB2BGR)


def visualize_coco_json(
    pred_json_path,
    gt_json_path,
    output_dir=None,
    image_base_path=None,
    max_images=None,
    mode="pred_vs_gt",
    alpha=0.5,
    score_threshold=None,
    display=False,
):
    """
    Utility function to visualize COCO format JSON annotations.

    Args:
        pred_json_path (str): Path to predictions JSON file
        gt_json_path (str): Path to ground truth JSON file
        output_dir (str): Directory to save output images. If None, only displays.
        image_base_path (str): Base path for images. If provided, used to compute full paths.
        max_images (int): Maximum number of images to visualize
        mode (str): One of "predictions_vs_gt", "predictions_only", "gt_only"
        alpha (float): Transparency level for overlays
        score_threshold (float): Minimum prediction score to visualize
        display (bool): Whether to display images in window

    Returns:
        dict: Summary of visualized images
    """

    # Load JSON files
    with open(pred_json_path) as f:
        pred_data_raw = json.load(f)
    with open(gt_json_path) as f:
        gt_data = json.load(f)

    # Convert predictions list to COCO format if needed
    if isinstance(pred_data_raw, list):
        # Predictions are in detection results format, convert to COCO format
        pred_data = {
            "images": gt_data.get("images", []),
            "annotations": pred_data_raw,
            "categories": gt_data.get("categories", []),
        }
    else:
        pred_data = pred_data_raw

    logger = setup_logger()
    logger.info(f"Loaded predictions from {pred_json_path}")
    logger.info(f"Loaded GT from {gt_json_path}")

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get list of images
    images = gt_data.get("images", [])
    if max_images:
        images = images[:max_images]

    summary = {
        "total_images": len(images),
        "visualized_images": 0,
        "failed_images": 0,
        "errors": [],
    }

    for img_info in tqdm.tqdm(images, desc="Visualizing"):
        try:
            img_id = img_info["id"]
            file_name = img_info["file_name"]

            # Construct full image path
            if image_base_path:
                image_path = os.path.join(image_base_path, file_name)
            else:
                image_path = file_name

            # Check if file exists
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                summary["failed_images"] += 1
                summary["errors"].append(f"File not found: {image_path}")
                continue

            # Create visualizer
            viz = COCOVisualizer(image_path, instance_mode="segmentation")

            # Draw based on mode
            if mode == "pred_vs_gt":
                viz.draw_predictions_vs_gt(
                    pred_data,
                    gt_data,
                    img_id,
                    alpha=alpha,
                    score_threshold=score_threshold,
                )
            elif mode == "pred_only":
                viz.draw_from_coco_json(
                    pred_data,
                    img_id,
                    pred_color=(1, 0, 0),
                    alpha=alpha,
                    score_threshold=score_threshold,
                )
            elif mode == "gt_only":
                viz.draw_from_coco_json(
                    gt_data, img_id, pred_color=(0, 1, 0), alpha=alpha
                )

            # Save if output directory specified
            if output_dir:
                output_filename = (
                    os.path.splitext(os.path.basename(file_name))[0] + "_viz.jpg"
                )
                output_path = os.path.join(output_dir, output_filename)
                viz.save_output(output_path)
                logger.info(f"Saved: {output_path}")

            # Display if requested
            if display:
                output_img = viz.get_image()
                cv2.imshow(WINDOW_NAME, output_img)
                key = cv2.waitKey(0)
                if key == 27:  # ESC to quit
                    break

            summary["visualized_images"] += 1

        except Exception as e:
            logger.error(
                f"Error visualizing image {img_info.get('file_name', 'unknown')}: {str(e)}"
            )
            summary["failed_images"] += 1
            summary["errors"].append(str(e))

    if display:
        cv2.destroyAllWindows()

    # Log summary
    logger.info("=" * 50)
    logger.info("Visualization Summary:")
    logger.info(f"Total images: {summary['total_images']}")
    logger.info(f"Successfully visualized: {summary['visualized_images']}")
    logger.info(f"Failed: {summary['failed_images']}")

    return summary


def coco_json_comparison_example(gt_json, pred_json, output_dir):
    """
    Example function showing how to compare predictions with ground truth.
    """
    visualize_coco_json(
        pred_json_path=pred_json,
        gt_json_path=gt_json,
        output_dir=output_dir,
        mode="predictions_vs_gt",
        alpha=0.5,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="COCO Polygon Visualizer")

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    # Arguments for COCO JSON visualization
    parser.add_argument(
        "--pred-json",
        help="Path to predictions JSON file in COCO format",
    )
    parser.add_argument(
        "--gt-json",
        help="Path to ground truth JSON file in COCO format",
    )
    parser.add_argument(
        "--image-base-path",
        help="Base path for images referenced in JSON files",
    )
    parser.add_argument(
        "--coco-mode",
        default="pred_vs_gt",
        choices=["pred_vs_gt", "pred_only", "gt_only"],
        help="Visualization mode for COCO JSON data",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to visualize",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Transparency level for overlays (0.0-1.0)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Minimum prediction score to visualize. If omitted, no score filtering is applied.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display images in window",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # Handle COCO JSON visualization
    if args.pred_json and args.gt_json:
        logger.info("Starting COCO JSON visualization...")
        summary = visualize_coco_json(
            pred_json_path=args.pred_json,
            gt_json_path=args.gt_json,
            output_dir=args.output if args.output else None,
            image_base_path=args.image_base_path,
            max_images=args.max_images,
            mode=args.coco_mode,
            alpha=args.alpha,
            score_threshold=args.score_threshold,
            display=args.display and args.output is None,
        )
        logger.info(f"Visualization complete. Summary: {summary}")
        exit(0)
    else:
        logger.error("Please provide both --pred-json and --gt-json arguments")
        exit(1)
