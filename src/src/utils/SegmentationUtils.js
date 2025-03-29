export default class SegmentationUtils {

    static async getAllPolygonsFromSegmentation(data, weights, width, height, realWidth, realHeight) {
        const threshold = 0.5;
        const binaryMasks = [];

        for (let channelIndex = 0; channelIndex < weights; channelIndex++) {
            const binaryMask = new Uint8Array(height * width);
            for (let i = 0; i < height; i++) {
                for (let j = 0; j < width; j++) {
                    const index = (channelIndex * height * width) + (i * width + j);
                    binaryMask[i * width + j] = data[index] > threshold ? 1 : 0;
                }
            }
            binaryMasks.push(binaryMask);
        }

        const polygons = [];
        const scaleX = realWidth / width;
        const scaleY = realHeight / height;
        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext('2d');

        for (let index = 0; index < binaryMasks.length; index++) {
            const imageData = ctx.createImageData(width, height);
            const mask = binaryMasks[index];
            
            for (let i = 0; i < mask.length; i++) {
                const pixelIndex = i * 4;
                const value = mask[i] * 255;
                imageData.data[pixelIndex] = value;     // R
                imageData.data[pixelIndex + 1] = value; // G
                imageData.data[pixelIndex + 2] = value; // B
                imageData.data[pixelIndex + 3] = 255;   // A
            }
            
            ctx.putImageData(imageData, 0, 0);
            const path = new Path2D();
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    if (mask[y * width + x] === 1) {
                        path.rect(x, y, 1, 1);
                    }
                }
            }

            const classPolygons = [];
            ctx.save();
            ctx.beginPath();
            ctx.stroke(path);
            const pathPoints = [];
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    if (mask[y * width + x] === 1) {
                        const isEdge = (
                            x === 0 || x === width - 1 || y === 0 || y === height - 1 ||
                            mask[y * width + (x - 1)] === 0 ||
                            mask[y * width + (x + 1)] === 0 ||
                            mask[(y - 1) * width + x] === 0 ||
                            mask[(y + 1) * width + x] === 0
                        );
                        
                        if (isEdge) {
                            pathPoints.push({ x, y });
                        }
                    }
                }
            }
            
            if (pathPoints.length > 0) {
                const orderedPoints = [pathPoints[0]];
                const remaining = new Set(pathPoints.slice(1));
                
                while (remaining.size > 0) {
                    const current = orderedPoints[orderedPoints.length - 1];
                    let nearest = null;
                    let minDist = Infinity;
                    
                    for (const point of remaining) {
                        const dist = Math.hypot(point.x - current.x, point.y - current.y);
                        if (dist < minDist) {
                            minDist = dist;
                            nearest = point;
                        }
                    }
                    
                    if (nearest) {
                        orderedPoints.push(nearest);
                        remaining.delete(nearest);
                    } else {
                        break;
                    }
                }

                const scaledPoints = orderedPoints.map(point => ({
                    x: point.x * scaleX,
                    y: point.y * scaleY
                }));
                
                classPolygons.push(scaledPoints);
            }
            
            ctx.restore();
            polygons.push(classPolygons);
        }

        return polygons;
    }

    static buildRawSegmentationPolygon(polygonGroups) {
        const points = polygonGroups.filter(group => group.length > 3);
        return points;
    }

    static buildSegmentationPolygon(polygonGroups) {
        polygonGroups = polygonGroups.filter(group => group.length > 3);
        const flattenedPoints = polygonGroups.flat();
        const topLeft = flattenedPoints.reduce((min, curr) => {
            const score = curr.x + curr.y;
            const minScore = min.x + min.y;
            return score < minScore ? curr : min;
        });

        const topRight = flattenedPoints.reduce((max, curr) => {
            const score = -curr.x + curr.y;
            const maxScore = -max.x + max.y;
            return score < maxScore ? curr : max;
        });

        const bottomRight = flattenedPoints.reduce((max, curr) => {
            const score = curr.x + curr.y;
            const maxScore = max.x + max.y;
            return score > maxScore ? curr : max;
        });

        const bottomLeft = flattenedPoints.reduce((max, curr) => {
            const score = curr.x - curr.y;
            const maxScore = max.x - max.y;
            return score < maxScore ? curr : max;
        });

        const width = Math.abs(topRight.x - topLeft.x);
        const height = Math.abs(bottomLeft.y - topLeft.y);
        const margin = Math.min(width, height) * 0.00;

        const coordinates = [
            { x: topLeft.x - margin, y: topLeft.y - margin },
            { x: topRight.x + margin, y: topRight.y - margin },
            { x: bottomRight.x + margin, y: bottomRight.y + margin },
            { x: bottomLeft.x - margin, y: bottomLeft.y + margin }
        ];

        coordinates.forEach(coordinate => {
            if (coordinate.x < 0) {
                coordinate.x = 0;
            }

            if (coordinate.y < 0) {
                coordinate.y = 0;
            }
        });

        return coordinates;
    }

    static getSegmentationBoundingBox(mask) {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const points of mask) {
            for (const point of points) {
                const realX = point.x;
                const realY = point.y;

                if (realX < minX) {
                    minX = realX;
                }

                if (realY < minY) {
                    minY = realY;
                }

                if (realX > maxX) {
                    maxX = realX;
                }

                if (realY > maxY) {
                    maxY = realY;
                }
            }
        }

        return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
    }

    static calculateRectIntersect(predRect, segRect) {
        const predX2 = predRect.x + predRect.w;
        const predY2 = predRect.y + predRect.h;
        const segX2 = segRect.x + segRect.w;
        const segY2 = segRect.y + segRect.h;
        const x1 = Math.max(predRect.x, segRect.x);
        const y1 = Math.max(predRect.y, segRect.y);
        const x2 = Math.min(predX2, segX2);
        const y2 = Math.min(predY2, segY2);
        const intersectionWidth = Math.max(0, x2 - x1);
        const intersectionHeight = Math.max(0, y2 - y1);
        const intersectionArea = intersectionWidth * intersectionHeight;
        const segArea = segRect.w * segRect.h;
        const percentage = segArea > 0 ? (intersectionArea / segArea) : 0;
        return percentage;
    }

    static computePolygonClassMaxArea(polygonClasses) {
        let maxArea = 0;
        for (let polygonClass of polygonClasses) {
            const area = SegmentationUtils.computePolygonArea(polygonClass)
            if (maxArea == 0 || area > maxArea) {
                maxArea = area;
            }
        }

        return maxArea * 4;
    }

    static computePolygonArea(polygon) {
        if (!polygon || polygon.length < 3) {
            return 0;
        }

        let area = 0;

        for (let i = 0; i < polygon.length; i++) {
            const p1 = polygon[i];
            const p2 = polygon[(i + 1) % polygon.length];
            area += (p1.x * p2.y) - (p1.y * p2.x);
        }

        return Math.abs(area / 2);
    }
}
