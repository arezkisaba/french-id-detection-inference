import * as Constants from "../common/constants.js";

export class ImageUtils {

    static rotateBase64Image(base64, quality) {
        return new Promise((resolve, reject) => {
            let image = new Image();

            image.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = image.height;
                canvas.height = image.width;

                const ctx = canvas.getContext('2d');
                ctx.translate(image.height / 2, image.width / 2);
                ctx.rotate(Math.PI / 2);
                ctx.drawImage(image, - image.width / 2, - image.height / 2);

                resolve(canvas.toDataURL("image/jpeg", quality));
            };

            image.onerror = reject;
            image.src = base64;
        });
    }

    static resizeImageByMaxSize(base64, maxWidth, maxHeight, quality, rounds = 2) {
        return new Promise((resolve, reject) => {
            let image = new Image();

            image.onload = () => {
                const canvas = document.createElement('canvas');

                canvas.width = image.width;
                canvas.height = image.height;

                const ctx = canvas.getContext('2d');

                ctx.imageSmoothingEnabled = false;
                if (typeof ctx.webkitImageSmoothingEnabled != 'undefined') {
                    ctx.webkitImageSmoothingEnabled = false;
                }
                if (typeof ctx.mozImageSmoothingEnabled != 'undefined') {
                    ctx.mozImageSmoothingEnabled = false;
                }

                ctx.drawImage(image, 0, 0);

                let ratio1 = maxWidth / image.width;
                let ratio2 = maxHeight / image.height;

                ImageUtils.resizeImageByRatio(image, canvas, ratio1 < ratio2 ? ratio1 : ratio2);

                resolve(canvas.toDataURL("image/jpeg", quality, rounds));
            }

            image.onerror = reject;

            image.src = base64;
        });
    }

    static resizeImageByRatio(img, canvas, ratio, rounds) {
        if (ratio === 1) {
            return canvas;
        }

        let canvasContext = canvas.getContext("2d");
        let canvasCopy = document.createElement("canvas");
        let copyContext = canvasCopy.getContext("2d");
        let canvasCopy2 = document.createElement("canvas");
        let copyContext2 = canvasCopy2.getContext("2d");

        let imgWidth = img.width;
        let imgHeight = img.height;

        canvasCopy.width = imgWidth;
        canvasCopy.height = imgHeight;
        copyContext.drawImage(img, 0, 0);

        canvasCopy2.width = imgWidth;
        canvasCopy2.height = imgHeight;
        copyContext2.drawImage(canvasCopy, 0, 0, canvasCopy.width, canvasCopy.height, 0, 0, canvasCopy2.width, canvasCopy2.height);

        for (let i = 1; i <= rounds; i++) {
            let roundRatio = ratio > 1 ? ((ratio - 1) / rounds * i) + 1 : 1 - ((1 - ratio) / rounds * i);
            let targetedWidth = imgWidth * roundRatio / i;
            let targetedHeight = imgHeight * roundRatio / i;
            canvasCopy.width = targetedWidth;
            canvasCopy.height = targetedHeight;
            copyContext.drawImage(canvasCopy2, 0, 0, canvasCopy2.width, canvasCopy2.height, 0, 0, canvasCopy.width, canvasCopy.height);
            canvasCopy2.width = targetedWidth;
            canvasCopy2.height = targetedHeight;
            copyContext2.drawImage(canvasCopy, 0, 0, canvasCopy.width, canvasCopy.height, 0, 0, canvasCopy2.width, canvasCopy2.height);

        }

        canvas.width = imgWidth * ratio;
        canvas.height = imgHeight * ratio;
        canvasContext.drawImage(canvasCopy2, 0, 0, canvasCopy2.width, canvasCopy2.height, 0, 0, canvas.width, canvas.height);

        return canvas;
    }

    static base64ToImageData(base64, modelWidth, modelHeight) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = base64;
            img.onload = function () {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = modelWidth;
                canvas.height = modelHeight;
                ctx.drawImage(img, 0, 0, modelWidth, modelHeight);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                resolve(imageData);
            };

            img.onerror = function () {
                reject(new Error('Failed to load image from Base64 string.'));
            };
        });
    }

    static async createCanvasFromBase64(base64) {
        const img = await ImageUtils.createImgFromBase64(base64);
        const canvas = ImageUtils.createCanvasFromImg(img);
        return canvas;
    }

    static createImgFromBase64(base64String) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = base64String;
            img.onload = function () {
                resolve(img);
            };
            img.onerror = function (err) {
                reject(err);
            };
        });
    }

    static createCanvasFromImg(img) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        context.drawImage(img, 0, 0, img.width, img.height);
        return canvas;
    }

    static convertVideoToImageData(video, modelWidth, modelHeight) {
        const canvas = document.createElement("canvas");
        canvas.width = modelWidth;
        canvas.height = modelHeight;
        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        return context.getImageData(0, 0, canvas.width, canvas.height);
    }

    static convertVideoFrameToBase64(videoElement) {
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg');
    }

    static convertImageDataToFloat32(imageData, config) {
        const batchSize = config.batchSize;
        const channels = config.channels;
        const modelWidth = config.width;
        const modelHeight = config.height;
        const input = new Float32Array(batchSize * channels * modelWidth * modelHeight);
        for (let i = 0; i < imageData.data.length; i += 4) {
            const index = Math.floor(i / 4);
            const row = Math.floor(index / modelWidth);
            const col = index % modelWidth;
            input[row * modelWidth + col] = imageData.data[i] / 255.0; // R
            input[modelWidth * modelHeight + row * modelWidth + col] = imageData.data[i + 1] / 255.0; // G
            input[2 * modelWidth * modelHeight + row * modelWidth + col] = imageData.data[i + 2] / 255.0; // B
        }
        return input;
    }

    static cropBase64Image(base64String, x, y, width, height) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = base64String;
            img.onload = function () {
                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, x, y, width, height, 0, 0, width, height);
                const croppedBase64 = canvas.toDataURL();
                resolve(croppedBase64);
                // resolve(enhanceQualityWithJimp(croppedBase64));
            };

            img.onerror = function () {
                reject(new Error('Failed to load image from Base64 string.'));
            };
        });
    }

    static cropImageToPolygon(base64Image, coordinates, angle) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = base64Image;
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                const newCoordinates = coordinates.map(coord => ({ x: coord.x, y: coord.y }));
                const minX = Math.min(...newCoordinates.map(coord => coord.x));
                const maxX = Math.max(...newCoordinates.map(coord => coord.x));
                const minY = Math.min(...newCoordinates.map(coord => coord.y));
                const maxY = Math.max(...newCoordinates.map(coord => coord.y));
                canvas.width = maxX - minX;
                canvas.height = maxY - minY;
                ctx.translate(-minX, -minY);
                ctx.beginPath();
                ctx.moveTo(newCoordinates[0].x, newCoordinates[0].y);
                for (let i = 1; i < newCoordinates.length; i++) {
                    ctx.lineTo(newCoordinates[i].x, newCoordinates[i].y);
                }
                ctx.closePath();
                ctx.clip();
                ctx.drawImage(img, 0, 0);

                const centerX = (minX + maxX) / 2;
                const centerY = (minY + maxY) / 2;
                const rotatedCanvas = document.createElement('canvas');
                const rotatedCtx = rotatedCanvas.getContext('2d');
                rotatedCanvas.width = canvas.width;
                rotatedCanvas.height = canvas.height;
                rotatedCtx.translate(rotatedCanvas.width / 2, rotatedCanvas.height / 2);
                rotatedCtx.rotate((angle * Math.PI) / 180);
                rotatedCtx.translate(-centerX + minX, -centerY + minY);
                rotatedCtx.drawImage(canvas, 0, 0);

                const croppedBase64 = rotatedCanvas.toDataURL('image/png');
                resolve(croppedBase64);
            };
            img.onerror = (error) => {
                reject(error);
            };
        });
    }

    static cropImageWithAngleCorrection(base64Image, seg, outputCanvas) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = base64Image;
            img.onload = () => {
                const rotatedCtx = outputCanvas.getContext('2d');
                if (!seg) {
                    rotatedCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
                    resolve();
                    return;
                }
    
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                const newCoordinates = seg.polygon.map(coord => ({ x: coord.x, y: coord.y }));
                const minX = Math.min(...newCoordinates.map(coord => coord.x));
                const maxX = Math.max(...newCoordinates.map(coord => coord.x));
                const minY = Math.min(...newCoordinates.map(coord => coord.y));
                const maxY = Math.max(...newCoordinates.map(coord => coord.y));
                canvas.width = maxX - minX;
                canvas.height = maxY - minY;
                ctx.translate(-minX, -minY);
                ctx.beginPath();
                ctx.moveTo(newCoordinates[0].x, newCoordinates[0].y);
                for (let i = 1; i < newCoordinates.length; i++) {
                    ctx.lineTo(newCoordinates[i].x, newCoordinates[i].y);
                }
                ctx.closePath();
                ctx.clip();
                ctx.drawImage(img, 0, 0);

                const centerX = (minX + maxX) / 2;
                const centerY = (minY + maxY) / 2;
                outputCanvas.width = canvas.width;
                outputCanvas.height = canvas.height;
                rotatedCtx.translate(outputCanvas.width / 2, outputCanvas.height / 2);
                rotatedCtx.rotate((seg.angle * Math.PI) / 180);
                rotatedCtx.translate(-centerX + minX, -centerY + minY);
                rotatedCtx.drawImage(canvas, 0, 0);
                resolve();
            };
            img.onerror = (error) => {
                reject(error);
            };
        });
    }

    static async enhanceQualityWithJimp(inputBase64) {
        try {
            const jimpImage = await Jimp.read(inputBase64);
            jimpImage
                .greyscale()
                .contrast(0.1)
                .brightness(0.50)
                .normalize()
                .blur(1)
                .convolute([
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]
                ])
                .threshold({ max: 200 });
            const { width, height, data } = jimpImage.bitmap;
            const uint8ClampedArray = new Uint8ClampedArray(data);
            const imageData = new ImageData(uint8ClampedArray, width, height);
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            ctx.putImageData(imageData, 0, 0);
            return canvas.toDataURL();
        } catch (err) {
            throw new Error(`Error processing image: ${err}`);
        }
    }

    static getImageDimensions(base64String) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = base64String;
            img.onload = function () {
                resolve({ width: this.width, height: this.height });
            };

            img.onerror = function () {
                reject(new Error('Failed to load image from Base64 string.'));
            };
        });
    }

    static getImageDimensions(uiElement) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.src = base64String;
            img.onload = function () {
                resolve({ width: this.width, height: this.height });
            };

            img.onerror = function () {
                reject(new Error('Failed to load image from Base64 string.'));
            };
        });
    }

    static calculateRotation(coordinates) {
        function getObjectWithMaxY(points) {
            let maxPoint = points[0];
            for (let i = 1; i < points.length; i++) {
                if (points[i].y > maxPoint.y) {
                    maxPoint = points[i];
                }
            }

            return maxPoint;
        }

        const coordinatesToTake = [coordinates[2], coordinates[3]];
        const commonPoint = getObjectWithMaxY(coordinatesToTake);
        let otherPoint = undefined;
        if (commonPoint.x === coordinates[2].x && commonPoint.y === coordinates[2].y) {
            otherPoint = coordinates[3];
        } else if (commonPoint.x === coordinates[3].x && commonPoint.y === coordinates[3].y) {
            otherPoint = coordinates[2];
        }

        let segments = [];
        let newImaginaryPointForAngleCalculation = {};
        if (commonPoint.x === coordinates[2].x && commonPoint.y === coordinates[2].y) {
            newImaginaryPointForAngleCalculation.x = coordinates[3].x;
            newImaginaryPointForAngleCalculation.y = commonPoint.y;
        } else if (commonPoint.x === coordinates[3].x && commonPoint.y === coordinates[3].y) {
            newImaginaryPointForAngleCalculation.x = coordinates[2].x;
            newImaginaryPointForAngleCalculation.y = commonPoint.y;
        }

        segments.push([otherPoint, commonPoint]);
        segments.push([commonPoint, newImaginaryPointForAngleCalculation]);

        const dx1 = otherPoint.x - commonPoint.x;
        const dy1 = otherPoint.y - commonPoint.y;
        const dx2 = newImaginaryPointForAngleCalculation.x - commonPoint.x;
        const dy2 = newImaginaryPointForAngleCalculation.y - commonPoint.y;
        const dotProduct = dx1 * dx2 + dy1 * dy2;
        const magnitude1 = Math.sqrt(dx1 * dx1 + dy1 * dy1);
        const magnitude2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
        const cosTheta = dotProduct / (magnitude1 * magnitude2);
        const angleRad = Math.acos(cosTheta);
        let angleDeg = (angleRad * 180) / Math.PI;

        if (commonPoint.x === coordinates[2].x && commonPoint.y === coordinates[2].y) {
            angleDeg *= -1;
        }

        return angleDeg;
    }

    static calculateMotionPercentage(currentFrame, previousFrame) {
        let diffPixels = 0;
        for (let i = 0; i < currentFrame.data.length; i += 4) {
            const diffR = Math.abs(currentFrame.data[i] - previousFrame.data[i]);
            const diffG = Math.abs(currentFrame.data[i + 1] - previousFrame.data[i + 1]); 
            const diffB = Math.abs(currentFrame.data[i + 2] - previousFrame.data[i + 2]);
            const totalDiff = diffR + diffG + diffB;
            
            if (totalDiff > 100) {
                diffPixels++;
            }
        }

        return (diffPixels / (currentFrame.width * currentFrame.height));
    }
}
