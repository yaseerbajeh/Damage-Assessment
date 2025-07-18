import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
import cv2
import numpy as np
import logging
import base64
from io import BytesIO
import asyncio
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Centroid(BaseModel):
    centroid_id: int
    num_pixels: int
    centroid_rgb: List[float]
    mean_red: float
    mean_green: float
    mean_blue: float

class DamageSummary(BaseModel):
    details: str

class Fruit(BaseModel):
    before_clusters: List[Centroid]
    after_clusters: List[Centroid]
    before_clustered_image: str
    after_clustered_image: str
    diff_image: str
    damage_summary: DamageSummary
    before_detected_image: str  # Added for images with bounding boxes
    after_detected_image: str   # Added for images with bounding boxes

class Fruits(BaseModel):
    fruits: List[Fruit]

app = FastAPI(debug=True)

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_db = {"fruits": []}

model = YOLO(r'C:\Users\yass1\PycharmProjects\fastApiProject - Copy\backend\dataset\runs\detect\train2\weights\best.pt')

def detect_buildings(image_content) -> Tuple[List[Tuple[int, int, int, int]], int, int]:
    """Hybrid detection using YOLOv8 and clustering to identify buildings."""
    logger.info("Starting hybrid building detection with YOLOv8 and clustering")

    nparr = np.frombuffer(image_content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Failed to decode image for detection")
        raise ValueError("Failed to decode image")

    height, width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    logger.info("Running YOLOv8 detection")
    results = model.predict(img, conf=0.5)
    yolo_boxes = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                yolo_boxes.append((x1, y1, x2, y2))
    yolo_count = len(yolo_boxes)
    logger.info(f"YOLOv8 detected {yolo_count} buildings")

    mask = np.ones((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in yolo_boxes:
        mask[y1:y2, x1:x2] = 0

    logger.info("Clustering remaining regions")
    pixels = img_rgb.reshape(-1, 3).astype(np.float64)
    remaining_pixels = pixels[mask.reshape(-1) == 1]

    if len(remaining_pixels) < 100:
        logger.info("Too few pixels remain for clustering; using YOLOv8 boxes only")
        return yolo_boxes, yolo_count, 0

    k = 8
    np.random.seed(42)
    centroids = np.random.uniform(0, 255, size=(k, 3)).astype(np.float64)
    centroids.setflags(write=True)

    max_iter = 5
    for iter in range(max_iter):
        distances = np.sqrt(np.sum((centroids[:, np.newaxis, :] - remaining_pixels[np.newaxis, :, :]) ** 2, axis=2))
        assignment = np.argmin(distances, axis=0)

        prev_centroids = np.copy(centroids)
        for i in range(k):
            ind = np.where(assignment == i)[0]
            if len(ind) > 0:
                centroids[i, :] = np.mean(remaining_pixels[ind], axis=0)
            else:
                np.random.seed(42 + i)
                centroids[i, :] = np.random.uniform(0, 255, size=3).astype(np.float64)

        if np.allclose(centroids, prev_centroids):
            break

    cluster_boxes = []
    full_assignment = np.full((height * width,), -1, dtype=int)
    idx = 0
    for i in range(height * width):
        if mask.reshape(-1)[i] == 1:
            full_assignment[i] = assignment[idx]
            idx += 1

    for i in range(k):
        r, g, b = centroids[i]
        is_white = (150 <= r <= 255) and (150 <= g <= 255) and (150 <= b <= 255)
        cluster_pixels = np.where(full_assignment == i)[0]
        num_pixels = len(cluster_pixels)
        if is_white and num_pixels >= 100:
            coords = np.unravel_index(cluster_pixels, (height, width))
            y_coords, x_coords = coords[0], coords[1]
            if len(x_coords) > 0 and len(y_coords) > 0:
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                if x2 > x1 and y2 > y1:
                    cluster_boxes.append((x1, y1, x2, y2))
                    logger.info(f"Cluster {i} detected as building: {num_pixels} pixels, box ({x1}, {y1}, {x2}, {y2})")
    cluster_count = len(cluster_boxes)

    all_boxes = yolo_boxes + cluster_boxes
    if not all_boxes:
        logger.info("No buildings detected by either method")
        return [], yolo_count, cluster_count

    final_boxes = []
    for box in all_boxes:
        if not final_boxes:
            final_boxes.append(box)
            continue
        overlap = False
        for existing_box in final_boxes:
            iou = calculate_iou(box, existing_box)
            if iou > 0.3:
                overlap = True
                break
        if not overlap:
            final_boxes.append(box)

    logger.info(f"Final detection: {len(final_boxes)} buildings after merging")
    return final_boxes, yolo_count, cluster_count

def draw_bounding_boxes(image_content, boxes, label="Building"):
    """Draw bounding boxes on the image and return as base64."""
    nparr = np.frombuffer(image_content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Failed to decode image for drawing boxes")
        raise ValueError("Failed to decode image")

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', img)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def cluster_image(image_content, bounding_boxes):
    logger.info("Starting image clustering with bounding boxes")
    try:
        nparr = np.frombuffer(image_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            raise ValueError("Failed to decode image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]

        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, y1, x2, y2 in bounding_boxes:
            mask[y1:y2, x1:x2] = 1
        pixels = img_rgb.reshape(-1, 3)[mask.reshape(-1) == 1].astype(np.float64)

        if len(pixels) == 0:
            logger.warning("No pixels detected within bounding boxes")
            pixels = img_rgb.reshape(-1, 3).astype(np.float64)

        logger.info(f"Processing {len(pixels)} pixels from building regions")

        k = 8
        np.random.seed(42)
        centroids = np.random.uniform(0, 255, size=(k, 3)).astype(np.float64)
        centroids.setflags(write=True)
        logger.info("Initialized centroids")

        max_iter = 5
        moved = True
        iter = 0

        while moved and iter < max_iter:
            iter += 1
            logger.info(f"Iteration {iter}")

            distances = np.sqrt(np.sum((centroids[:, np.newaxis, :] - pixels[np.newaxis, :, :]) ** 2, axis=2))
            assignment = np.argmin(distances, axis=0)

            prev_centroids = np.copy(centroids)

            for i in range(k):
                ind = np.where(assignment == i)[0]
                if len(ind) > 0:
                    centroids[i, :] = np.mean(pixels[ind], axis=0)
                else:
                    np.random.seed(42 + i)
                    centroids[i, :] = np.random.uniform(0, 255, size=3).astype(np.float64)

            if np.allclose(centroids, prev_centroids):
                moved = False

        logger.info("Clustering complete")
        logger.info("Generating clustered image")
        all_pixels = img_rgb.reshape(-1, 3).astype(np.float64)
        distances = np.sqrt(np.sum((centroids[:, np.newaxis, :] - all_pixels[np.newaxis, :, :]) ** 2, axis=2))
        full_assignment = np.argmin(distances, axis=0)

        clustered_img = np.zeros_like(all_pixels)
        for i in range(k):
            clustered_img[full_assignment == i] = centroids[i]
        clustered_img = clustered_img.reshape(height, width, 3).astype(np.uint8)

        clustered_img_bgr = cv2.cvtColor(clustered_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', clustered_img_bgr)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        logger.info("Clustered image encoded as base64")

        clusters = []
        for i in range(k):
            ind = np.where(assignment == i)[0]
            num_pixels = len(ind) * (width * height // len(pixels)) if len(pixels) > 0 else 0
            if len(ind) > 0:
                mean_rgb = np.mean(pixels[ind], axis=0)
                mean_red, mean_green, mean_blue = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
            else:
                mean_red, mean_green, mean_blue = 0.0, 0.0, 0.0
            clusters.append(Centroid(
                centroid_id=i,
                num_pixels=num_pixels,
                centroid_rgb=[float(centroids[i, 0]), float(centroids[i, 1]), float(centroids[i, 2])],
                mean_red=mean_red,
                mean_green=mean_green,
                mean_blue=mean_blue
            ))
            logger.info(f"Cluster {i}: {num_pixels:,} pixels, RGB {centroids[i]}")

        return clusters, base64_image, full_assignment, all_pixels, img_rgb
    except Exception as e:
        logger.error(f"Error in cluster_image: {str(e)}")
        raise

def assess_damage(before_clusters, after_clusters, before_assignment, after_assignment, before_pixels, after_pixels,
                  height, width, before_yolo_count, before_cluster_count, after_yolo_count, after_cluster_count):
    logger.info("Assessing damage")
    try:
        logger.info("Before Clusters:")
        for centroid in before_clusters:
            logger.info(f"Cluster {centroid.centroid_id}: RGB {centroid.centroid_rgb}, Pixels: {centroid.num_pixels:,}")
        logger.info("After Clusters:")
        for centroid in after_clusters:
            logger.info(f"Cluster {centroid.centroid_id}: RGB {centroid.centroid_rgb}, Pixels: {centroid.num_pixels:,}")

        def is_white_building(centroid):
            r, g, b = centroid.centroid_rgb
            min_pixels = 100
            is_white = (150 <= r <= 255) and (150 <= g <= 255) and (150 <= b <= 255)
            return is_white and centroid.num_pixels >= min_pixels

        before_building_pixels = 0
        after_building_pixels = 0

        for centroid in before_clusters:
            if is_white_building(centroid):
                before_building_pixels += centroid.num_pixels
                logger.info(f"White building cluster detected (before): RGB {centroid.centroid_rgb}, {centroid.num_pixels:,} pixels")

        for centroid in after_clusters:
            if is_white_building(centroid):
                after_building_pixels += centroid.num_pixels
                logger.info(f"White building cluster detected (after): RGB {centroid.centroid_rgb}, {centroid.num_pixels:,} pixels")

        changed_pixels = max(0, before_building_pixels - after_building_pixels)

        diff_img = np.zeros_like(before_pixels)
        before_is_building = np.array([is_white_building(before_clusters[i]) for i in before_assignment])
        after_is_building = np.array([is_white_building(after_clusters[i]) for i in after_assignment])
        changed = before_is_building & ~after_is_building
        diff_img[changed] = [255, 0, 0]
        diff_img = diff_img.reshape(height, width, 3).astype(np.uint8)
        diff_img_bgr = cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', diff_img_bgr)
        diff_base64 = base64.b64encode(buffer).decode('utf-8')
        logger.info("Difference image generated")

        details = (
            f"Before image had {before_building_pixels:,} pixels in white building clusters.\n"
            f"After image had {after_building_pixels:,} pixels in white building clusters.\n"
            f"Approximately {changed_pixels:,} pixels changed.\n"
            f"Before: YOLOv8 detected {before_yolo_count} buildings, Clustering detected {before_cluster_count} buildings.\n"
            f"After: YOLOv8 detected {after_yolo_count} buildings, Clustering detected {after_cluster_count} buildings."
        )

        return DamageSummary(details=details), diff_base64
    except Exception as e:
        logger.error(f"Error in assess_damage: {str(e)}")
        raise

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

@app.get("/fruits", response_model=Fruits)
def get_fruits():
    logger.info("Fetching stored clustering results")
    try:
        return Fruits(fruits=memory_db["fruits"])
    except Exception as e:
        logger.error(f"Error in GET /fruits: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch clustering data")

@app.delete("/fruits")
def delete_fruits():
    """Clear all stored clustering results."""
    logger.info("Deleting all stored clustering results")
    try:
        memory_db["fruits"] = []
        return {"message": "Clustering results deleted successfully"}
    except Exception as e:
        logger.error(f"Error in DELETE /fruits: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete clustering data")

@app.post("/images")
async def add_fruit(beforeImage: UploadFile = File(...), afterImage: UploadFile = File(...), skip_ssim: bool = False):
    logger.info("Received image upload request")
    try:
        if not beforeImage.content_type.startswith("image/") or not afterImage.content_type.startswith("image/"):
            logger.error("Invalid image file type")
            raise HTTPException(status_code=400, detail="Only image files are allowed")

        before_content = await beforeImage.read()
        after_content = await afterImage.read()

        logger.info("Checking image dimensions")
        before_nparr = np.frombuffer(before_content, np.uint8)
        after_nparr = np.frombuffer(after_content, np.uint8)
        before_img = cv2.imdecode(before_nparr, cv2.IMREAD_COLOR)
        after_img = cv2.imdecode(after_nparr, cv2.IMREAD_COLOR)
        if before_img is None or after_img is None:
            logger.error("Failed to decode images")
            raise HTTPException(status_code=400, detail="Failed to decode images")

        before_height, before_width = before_img.shape[:2]
        after_height, after_width = after_img.shape[:2]
        if (before_height, before_width) != (after_height, after_width):
            logger.info(f"Resizing after image from ({after_width}x{after_height}) to ({before_width}x{before_height})")
            after_img = cv2.resize(after_img, (before_width, before_height), interpolation=cv2.INTER_LINEAR)
            _, after_buffer = cv2.imencode('.jpg', after_img)
            after_content = after_buffer.tobytes()
            logger.info("After image resized and re-encoded")

        if not skip_ssim:
            logger.info("Validating image similarity")
            before_rgb = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
            after_rgb = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
            try:
                similarity = ssim(before_rgb, after_rgb, channel_axis=2)
                logger.info(f"SSIM similarity: {similarity:.2f}")
                if similarity < 0.2:
                    raise HTTPException(status_code=400, detail=f"Images appear to depict different scenes (SSIM {similarity:.2f} < 0.2)")
            except Exception as e:
                logger.error(f"SSIM computation failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to compute image similarity: {str(e)}")
        else:
            logger.info("SSIM validation skipped")

        before_boxes, before_yolo_count, before_cluster_count = detect_buildings(before_content)
        after_boxes, after_yolo_count, after_cluster_count = detect_buildings(after_content)

        before_detected_image = draw_bounding_boxes(before_content, before_boxes, "Building")
        after_detected_image = draw_bounding_boxes(after_content, after_boxes, "Building")

        logger.info("Clustering images in parallel")
        before_task = asyncio.to_thread(cluster_image, before_content, before_boxes)
        after_task = asyncio.to_thread(cluster_image, after_content, after_boxes)
        (before_clusters, before_clustered_image, before_assignment, before_pixels, before_img_rgb), \
            (after_clusters, after_clustered_image, after_assignment, after_pixels, _) = await asyncio.gather(before_task, after_task)

        logger.info("Performing damage assessment")
        height, width = before_img_rgb.shape[:2]
        try:
            damage_summary, diff_base64 = assess_damage(before_clusters, after_clusters, before_assignment,
                                                        after_assignment, before_pixels, after_pixels, height, width,
                                                        before_yolo_count, before_cluster_count, after_yolo_count, after_cluster_count)
        except Exception as e:
            logger.error(f"Damage assessment failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Damage assessment failed: {str(e)}")

        fruit = Fruit(
            before_clusters=before_clusters,
            after_clusters=after_clusters,
            before_clustered_image=before_clustered_image,
            after_clustered_image=after_clustered_image,
            diff_image=diff_base64,
            damage_summary=damage_summary,
            before_detected_image=before_detected_image,
            after_detected_image=after_detected_image
        )
        memory_db["fruits"] = []  # Clear previous results
        memory_db["fruits"].append(fruit)
        logger.info("Stored clustering results")

        return {
            "message": "Images clustered successfully",
            "before_clusters": before_clusters,
            "after_clusters": after_clusters,
            "before_clustered_image": before_clustered_image,
            "after_clustered_image": after_clustered_image,
            "diff_image": diff_base64,
            "damage_summary": damage_summary,
            "before_detected_image": before_detected_image,
            "after_detected_image": after_detected_image
        }
    except HTTPException as e:
        logger.error(f"HTTP error in add_fruit: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in add_fruit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error processing images: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)