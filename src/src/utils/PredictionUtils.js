import * as tf from "@tensorflow/tfjs";
import Tesseract from 'tesseract.js';
import customLabels from "../json/customLabels.json";
import * as Constants from "../common/constants.js";
import { ImageUtils } from "./ImageUtils.js";
import IndexedDBUtils from "./IndexedDBUtils.js";
import SegmentationUtils from "./SegmentationUtils.js";

export default class PredictionUtils {
    static #intersectThreshold = 0.99;
    static #probabilityThreshold = 0.25;
    static #yoloSeg;
    static #mobileNetModel;
    static #crnnModel;

    static async loadModels() {
        await Promise.all([
            PredictionUtils.loadYoloSeg(),
            // PredictionUtils.loadMobileNet(),
            // PredictionUtils.loadCrnn()
        ]);
    }

    static loadOpenCv() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = '/opencv.js';
            script.async = true;
            script.onload = () => {
                if (window.cv) {
                    console.log(`OpenCV loaded from '${script.src}'`);
                    resolve();
                } else {
                    reject(new Error('OpenCV not loaded'));
                }
            };
            script.onerror = () => reject(new Error('Failed to load OpenCV'));
            document.body.appendChild(script);
        });
    }

    static async loadYoloSeg() {
        const model = await PredictionUtils.loadGraphModelWithIndexedDB(Constants.DOC_SEG_TFJS_CONFIG.model.path);
        const dummyInput = tf.zeros([1, Constants.DOC_SEG_TFJS_CONFIG.model.height, Constants.DOC_SEG_TFJS_CONFIG.model.width, 3]);
        model.execute(dummyInput);
        PredictionUtils.#yoloSeg = model;
    }

    static async loadMobileNet() {
        const model = await PredictionUtils.loadGraphModelWithIndexedDB(Constants.TEXT_DET_TFJS_CONFIG.model.path);
        const dummyInput = tf.zeros([1, Constants.TEXT_DET_TFJS_CONFIG.model.height, Constants.TEXT_DET_TFJS_CONFIG.model.width, 3]);
        await model.executeAsync(dummyInput);
        PredictionUtils.#mobileNetModel = model;
    }

    static async loadCrnn() {
        const model = await PredictionUtils.loadGraphModelWithIndexedDB(Constants.TEXT_REC_TFJS_CONFIG.model.path);
        const dummyInput = tf.zeros([1, Constants.TEXT_REC_TFJS_CONFIG.model.height, Constants.TEXT_REC_TFJS_CONFIG.model.width, 3]);
        await model.executeAsync(dummyInput);
        PredictionUtils.#crnnModel = model;
    }

    static async getBestDocumentPrediction(base64, uiElement, realWidth, realHeight, addSegmentation) {
        const [modelHeight, modelWidth] = PredictionUtils.#yoloSeg.inputs[0].shape.slice(1, 3);
        if (base64) {
            const imageDataForModel = await ImageUtils.base64ToImageData(base64, modelWidth, modelHeight);
            return await PredictionUtils.getYoloSegPredictionsWithTfjs(imageDataForModel, realWidth, realHeight, modelWidth, modelHeight, addSegmentation);
        } else {
            const imageDataForModel = ImageUtils.convertVideoToImageData(uiElement, modelWidth, modelHeight);
            return await PredictionUtils.getYoloSegPredictionsWithTfjs(imageDataForModel, realWidth, realHeight, modelWidth, modelHeight, addSegmentation);
        }
    }

    static async getYoloSegPredictions(imageData, realWidth, realHeight, modelWidth, modelHeight, addSegmentation) {
        const inputTensor = tf.tidy(() => {
            const img = tf.browser.fromPixels(imageData);
            const [h, w] = img.shape.slice(0, 2);
            const maxSize = Math.max(w, h);
            const imgPadded = img.pad([
                [0, maxSize - h],
                [0, maxSize - w],
                [0, 0],
            ]);

            return tf.image
                .resizeBilinear(imgPadded, [modelWidth, modelHeight])
                .div(255.0)
                .expandDims(0);
        });
        let outputTensorDet = undefined;
        let outputTensorSeg = undefined;

        try {
            const outputPredictions = [];
            const results = PredictionUtils.#yoloSeg.execute(inputTensor);
            outputTensorDet = results[Object.keys(results)[0]];
            outputTensorSeg = tf.transpose(results[Object.keys(results)[1]], [0, 3, 1, 2]);
            const outputTensorDataDet = outputTensorDet.dataSync();
            const outputTensorDataSeg = outputTensorSeg.dataSync();
            const boxFeatureCount = 4;
            const scaleX = realWidth / modelWidth;
            const scaleY = realHeight / modelHeight;
            const featuresCountDet = outputTensorDet.shape[1];
            const predictionCountDet = outputTensorDet.shape[2];
            const weightMasksSeg = outputTensorSeg.shape[1];

            for (let predictionIndex = 0; predictionIndex < predictionCountDet; predictionIndex++) {
                const confidences = [];
                for (let i = boxFeatureCount; i < featuresCountDet - weightMasksSeg; i++) {
                    const confidenceClassIndex = predictionIndex + (i * predictionCountDet);
                    confidences.push(outputTensorDataDet[confidenceClassIndex]);
                }

                const score = Math.max(...confidences);
                if (score > PredictionUtils.#probabilityThreshold) {
                    const x_center = outputTensorDataDet[predictionIndex];
                    const y_center = outputTensorDataDet[predictionIndex + predictionCountDet];
                    const width = outputTensorDataDet[predictionIndex + 2 * predictionCountDet];
                    const height = outputTensorDataDet[predictionIndex + 3 * predictionCountDet];
                    const classIndex = confidences.indexOf(score);
                    const className = customLabels[classIndex];
                    outputPredictions.push({
                        className: className,
                        probability: (score * 100).toFixed(1),
                        rect: {
                            x: (x_center - width / 2) * scaleX,
                            y: (y_center - height / 2) * scaleY,
                            w: width * scaleX,
                            h: height * scaleY
                        },
                        seg: {}
                    });
                }
            }

            const sortedPredictions = outputPredictions.sort((a, b) => b.probability - a.probability);
            if (sortedPredictions.length === 0) {
                return undefined;
            }

            const prediction = sortedPredictions[0];
            if (addSegmentation) {
                prediction.seg = await PredictionUtils.getSegInformationsFromBinaryMask(
                    prediction,
                    outputTensorDataSeg,
                    outputTensorSeg.shape[1],
                    outputTensorSeg.shape[3],
                    outputTensorSeg.shape[2],
                    realWidth,
                    realHeight
                );
            }

            return prediction;
        } catch (error) {
            console.error("Erreur lors de la prédiction de detection TFJS :", error);
            throw error;
        } finally {
            tf.dispose([inputTensor, outputTensorDet, outputTensorSeg]);
        }
    }

    static async getYoloSegPredictionsWithTfjs(imageData, realWidth, realHeight, modelWidth, modelHeight, addSegmentation) {
        let globalScaleX, globalScaleY, xRatio, yRatio;
        const inputTensor = tf.tidy(() => {
            const img = tf.browser.fromPixels(imageData);
            const [h, w] = img.shape.slice(0, 2);
            const maxSize = Math.max(w, h);
            const imgPadded = img.pad([
                [0, maxSize - h],
                [0, maxSize - w],
                [0, 0],
            ]);

            xRatio = maxSize / w;
            yRatio = maxSize / h;
            globalScaleX = realWidth / w;
            globalScaleY = realHeight / h;
            return tf.image
                .resizeBilinear(imgPadded, [modelWidth, modelHeight])
                .div(255.0)
                .expandDims(0);
        });

        try {
            const outputPredictions = [];
            const results = PredictionUtils.#yoloSeg.execute(inputTensor);
            const transRes = tf.tidy(() => results[0].transpose([0, 2, 1]).squeeze());
            const transSegMask = tf.tidy(() => results[1].transpose([0, 3, 1, 2]).squeeze());
            const [modelSegHeight, modelSegWidth, modelSegChannel] = results[1].shape.slice(1);
            const boxes = tf.tidy(() => {
                const w = transRes.slice([0, 2], [-1, 1]);
                const h = transRes.slice([0, 3], [-1, 1]);
                const x1 = tf.sub(transRes.slice([0, 0], [-1, 1]), tf.div(w, 2));
                const y1 = tf.sub(transRes.slice([0, 1], [-1, 1]), tf.div(h, 2));
                return tf.concat([y1, x1, tf.add(y1, h), tf.add(x1, w)], 1).squeeze();
            });
            const [scores, classes] = tf.tidy(() => {
                const rawScores = transRes.slice([0, 4], [-1, customLabels.length]).squeeze();
                return [rawScores.max(1), rawScores.argMax(1)];
            });
            const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, PredictionUtils.#probabilityThreshold, 0.2);
            const detReady = tf.tidy(() =>
                tf.concat(
                    [
                        boxes.gather(nms, 0),
                        scores.gather(nms, 0).expandDims(1),
                        classes.gather(nms, 0).expandDims(1),
                    ],
                    1
                )
            );
            const masks = tf.tidy(() => {
                const sliced = transRes.slice([0, 4 + customLabels.length], [-1, modelSegChannel]).squeeze();
                return sliced
                    .gather(nms, 0)
                    .matMul(transSegMask.reshape([modelSegChannel, -1]))
                    .reshape([nms.shape[0], modelSegHeight, modelSegWidth]);
            });

            let overlay = tf.zeros([realHeight, realWidth, 4]);
            for (let i = 0; i < detReady.shape[0]; i++) {
                const rowData = detReady.slice([i, 0], [1, 6]);
                let [y1, x1, y2, x2, score, label] = rowData.dataSync();
                const upSampleBox = [
                    Math.floor(y1 * yRatio * globalScaleY),
                    Math.floor(x1 * xRatio * globalScaleX),
                    Math.round((y2 - y1) * yRatio * globalScaleY),
                    Math.round((x2 - x1) * xRatio * globalScaleX),
                ];

                if (addSegmentation) {
                    const downSampleBox = [
                        Math.floor((y1 * modelSegHeight) / modelHeight),
                        Math.floor((x1 * modelSegWidth) / modelWidth),
                        Math.round(((y2 - y1) * modelSegHeight) / modelHeight),
                        Math.round(((x2 - x1) * modelSegWidth) / modelWidth),
                    ];
                    const proto = tf.tidy(() => {
                        const sliced = masks.slice(
                            [
                                i,
                                downSampleBox[0] >= 0 ? downSampleBox[0] : 0,
                                downSampleBox[1] >= 0 ? downSampleBox[1] : 0,
                            ],
                            [
                                1,
                                downSampleBox[0] + downSampleBox[2] <= modelSegHeight ? downSampleBox[2] : modelSegHeight - downSampleBox[0],
                                downSampleBox[1] + downSampleBox[3] <= modelSegWidth ? downSampleBox[3] : modelSegWidth - downSampleBox[1],
                            ]
                        );
                        return sliced.squeeze().expandDims(-1);
                    });
                    const upsampleProto = tf.image.resizeBilinear(proto, [upSampleBox[2], upSampleBox[3]]);
                    const mask = tf.tidy(() => {
                        const padded = upsampleProto.pad([
                            [upSampleBox[0], realHeight - (upSampleBox[0] + upSampleBox[2])],
                            [upSampleBox[1], realWidth - (upSampleBox[1] + upSampleBox[3])],
                            [0, 0],
                        ]);
                        return padded.less(0.5);
                    });
                    overlay = tf.tidy(() => {
                        const newOverlay = overlay.where(mask, [...PredictionUtils.hexToRgba("#ff0000"), 150]);
                        overlay.dispose();
                        return newOverlay;
                    });
                    tf.dispose([proto, upsampleProto, mask]);
                }

                const [y, x, h, w] = upSampleBox;
                if (score > PredictionUtils.#probabilityThreshold) {
                    outputPredictions.push({
                        className: customLabels[label],
                        probability: (score * 100).toFixed(1),
                        rect: { x: x, y: y, w: w, h: h },
                        seg: {}
                    });
                }

                tf.dispose([rowData]);
            }

            const sortedPredictions = outputPredictions.sort((a, b) => b.probability - a.probability);
            if (sortedPredictions.length === 0) {
                return undefined;
            }

            const prediction = sortedPredictions[0];
            if (addSegmentation) {
                const overlayData = await overlay.data();
                // const polygonExtremities = PredictionUtils.findPolygonExtremities(overlayData, realWidth, realHeight);
                var clampedArray = new Uint8ClampedArray(overlayData);
                const maskImg = new ImageData(
                    clampedArray,
                    realWidth,
                    realHeight
                );
                prediction.maskImg = maskImg;
            }

            return prediction;
        } catch (error) {
            console.error("Erreur lors de la prédiction de detection TFJS :", error);
            throw error;
        } finally {
            // tf.dispose([inputTensor, outputTensorDet, outputTensorSeg]);
        }
    }

    static hexToRgba(hex, alpha) {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result
            ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
            : null;
    };

    static async getSegInformationsFromBinaryMask(prediction, data, weigths, modelWidth, modelHeight, realWidth, realHeight) {
        let possiblePolygons = [];
        const allShapes = await SegmentationUtils.getAllPolygonsFromSegmentation(
            data,
            weigths,
            modelHeight,
            modelWidth,
            realWidth,
            realHeight
        );
        for (const shape of allShapes) {
            const maskRect = SegmentationUtils.getSegmentationBoundingBox(shape);
            const intersectScore = SegmentationUtils.calculateRectIntersect(prediction.rect, maskRect);
            if (intersectScore < PredictionUtils.#intersectThreshold) {
                console.log("Intersect score too low:", intersectScore);
                continue;
            }

            const segmentationPolygon = SegmentationUtils.buildSegmentationPolygon(shape);
            possiblePolygons.push({ points: segmentationPolygon, intersectScore: intersectScore });
        }

        if (possiblePolygons.length === 0) {
            return undefined;
        }

        const bestPolygon = PredictionUtils.getBestPolygon(possiblePolygons, prediction.rect);
        return {
            possiblePolygons: possiblePolygons,
            polygon: bestPolygon,
            angle: ImageUtils.calculateRotation(bestPolygon)
        }
    }

    static getBestPolygon(possiblePolygons, rect) {
        if (!possiblePolygons || possiblePolygons.length === 0) {
            return undefined;
        }

        const calculateAngle = (v1, v2) => {
            const dot = v1.x * v2.x + v1.y * v2.y;
            const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
            const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
            const cosAngle = dot / (mag1 * mag2);
            return Math.acos(Math.min(Math.max(cosAngle, -1), 1)) * (180 / Math.PI);
        };

        const calculateRectangularity = (points) => {
            if (points.length < 4) {
                return 0;
            }

            // Réduire le polygone à 4 points en prenant les points les plus éloignés
            const simplifiedPoints = [];
            let maxDistance = 0;
            let p1, p2;

            // Trouver les deux points les plus éloignés
            for (let i = 0; i < points.length; i++) {
                for (let j = i + 1; j < points.length; j++) {
                    const dx = points[i].x - points[j].x;
                    const dy = points[i].y - points[j].y;
                    const distance = dx * dx + dy * dy;
                    if (distance > maxDistance) {
                        maxDistance = distance;
                        p1 = points[i];
                        p2 = points[j];
                    }
                }
            }

            simplifiedPoints.push(p1);

            // Trouver les deux points les plus éloignés de la ligne p1-p2
            let maxDist1 = 0;
            let maxDist2 = 0;
            let p3, p4;

            for (const point of points) {
                if (point === p1 || point === p2) continue;

                const distance = Math.abs(
                    (p2.y - p1.y) * point.x - (p2.x - p1.x) * point.y + p2.x * p1.y - p2.y * p1.x
                ) / Math.sqrt((p2.y - p1.y) * (p2.y - p1.y) + (p2.x - p1.x) * (p2.x - p1.x));

                if (distance > maxDist1) {
                    maxDist2 = maxDist1;
                    p4 = p3;
                    maxDist1 = distance;
                    p3 = point;
                } else if (distance > maxDist2) {
                    maxDist2 = distance;
                    p4 = point;
                }
            }

            simplifiedPoints.push(p3);
            simplifiedPoints.push(p2);
            simplifiedPoints.push(p4);

            // Calculer les angles entre les côtés adjacents
            let angles = [];
            for (let i = 0; i < 4; i++) {
                const current = simplifiedPoints[i];
                const next = simplifiedPoints[(i + 1) % 4];
                const prev = simplifiedPoints[(i + 3) % 4];

                const v1 = { x: next.x - current.x, y: next.y - current.y };
                const v2 = { x: prev.x - current.x, y: prev.y - current.y };
                angles.push(calculateAngle(v1, v2));
            }

            // Calculer la déviation par rapport à 90 degrés
            const angleDeviation = angles.reduce((sum, angle) => sum + Math.abs(angle - 90), 0) / 4;

            // Score de rectangularité (0 à 1, où 1 est parfaitement rectangulaire)
            const rectangularity = 1 - (angleDeviation / 90);
            return rectangularity;
        };

        return possiblePolygons
            .map(polygon => ({
                points: polygon.points,
                score: calculateRectangularity(polygon.points) * polygon.intersectScore
            }))
            .reduce((best, current) =>
                current.score > best.score ? current : best
                , { points: possiblePolygons[0].points, score: 0 })
            .points;
    }

    static async loadOnnxWithIndexedDB(modelUrl) {
        let cachedOnnx = await IndexedDBUtils.getFile(modelUrl);
        if (!cachedOnnx) {
            const response = await fetch(modelUrl);
            const buffer = await response.arrayBuffer();
            await IndexedDBUtils.saveFile(modelUrl, new Blob([buffer]));
            console.warn(`File '${modelUrl}' downloaded from network`);
        }

        cachedOnnx = await IndexedDBUtils.getFile(modelUrl);
        const arrayBuffer = await cachedOnnx.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        const model = await ort.InferenceSession.create(uint8Array);
        console.log(`File '${modelUrl}' loaded from cache`);
        return model;
    }

    static async loadGraphModelWithIndexedDB(modelUrl) {
        const modelKey = modelUrl;
        let cachedModelJson = await IndexedDBUtils.getFile(`${modelKey}/model.json`);

        let modelJson;
        if (!cachedModelJson) {
            const response = await fetch(modelUrl);
            modelJson = await response.json();
            await IndexedDBUtils.saveFile(`${modelKey}/model.json`, new Blob([JSON.stringify(modelJson)], { type: 'application/json' }));
            console.warn(`File '${modelUrl}' downloaded from network`);
        }

        cachedModelJson = await IndexedDBUtils.getFile(`${modelKey}/model.json`);
        const jsonText = await cachedModelJson.text();
        modelJson = JSON.parse(jsonText);
        console.log(`File '${modelUrl}' loaded from cache`);

        const weightFiles = modelJson.weightsManifest.flatMap(manifest =>
            manifest.paths.map(path => `${modelKey}/${path}`)
        );
        let cachedWeights = await Promise.all(weightFiles.map(async (weightFile) => {
            return await IndexedDBUtils.getFile(weightFile);
        }));

        const missingWeights = [];
        for (let i = 0; i < cachedWeights.length; i++) {
            if (!cachedWeights[i]) {
                missingWeights.push(weightFiles[i]);
            }
        }

        await Promise.all(missingWeights.map(async (weightFile) => {
            const realWeightFile = weightFile.replace('/model.json', '');
            const weightResponse = await fetch(realWeightFile);
            const weightBuffer = await weightResponse.arrayBuffer();
            await IndexedDBUtils.saveFile(weightFile, new Blob([weightBuffer]));
            console.warn(`File '${realWeightFile}' downloaded from network`);
        }));

        cachedWeights = await Promise.all(weightFiles.map(async (weightFile) => {
            return await IndexedDBUtils.getFile(weightFile);
        }));

        const filesToLoad = [
            new File([new Blob([JSON.stringify(modelJson)], { type: 'application/json' })], 'model.json')
        ];

        for (const cachedWeight of cachedWeights) {
            if (cachedWeight) {
                const buffer = await cachedWeight.arrayBuffer();
                const weightFile = cachedWeights.indexOf(cachedWeight);
                const realWeightFile = weightFiles[weightFile].replace('/model.json', '');
                const weightName = weightFiles[weightFile].split('/').pop();
                filesToLoad.push(new File([buffer], weightName));
                console.log(`File '${realWeightFile}' loaded from cache`);
            }
        }

        const model = await tf.loadGraphModel(tf.io.browserFiles(filesToLoad), {    
            onProgress: (progress) => {
                console.log(`Loading '${modelUrl}' : ${progress}%`);
            },
        });
        return model;
    }

    static async getOcrText(base64) {

        function sanitizeOcrText(ocrText) {
            return ocrText.replace(/[-_.,; ]/g, '').toLowerCase();
        }

        const ocrFuncs = [
            (base64) => PredictionUtils.getTextWithCrnn(base64),
            (base64) => PredictionUtils.getTextWithTesseract(base64)
        ]
        const ocrTexts = [];

        for (let i = 0; i < ocrFuncs.length; i++) {
            console.log(`Trying OCR method ${i + 1} of ${ocrFuncs.length}`);
            const ocrFunc = ocrFuncs[i];
            const ocrText = await ocrFunc(base64);
            const ocrTextSanitized = ocrText; //sanitizeOcrText(ocrText);
            console.log(`Text found : ${ocrTextSanitized}`);
            ocrTexts.push(ocrTextSanitized);
        }

        return ocrTexts;
    }

    static async getTextWithCrnn(base64) {
        const mobileNetPredictions = await PredictionUtils.getMobileNetPredictions(
            base64,
            Constants.TEXT_DET_TFJS_CONFIG.model.width,
            Constants.TEXT_DET_TFJS_CONFIG.model.height
        );
        const crnnPredictions = await PredictionUtils.getCrnnPredictions(
            mobileNetPredictions,
            [Constants.TEXT_REC_TFJS_CONFIG.model.height, Constants.TEXT_REC_TFJS_CONFIG.model.width]
        );

        let ocrText = "";
        for (let i = 0; i < crnnPredictions.length; i++) {
            ocrText += crnnPredictions[i];
        }
        return ocrText;
    }

    static async getTextWithTesseract(base64) {
        try {
            const worker = await Tesseract.createWorker('fra');
            const result = await worker.recognize(base64);
            await worker.terminate();
            return result.data.text;
        } catch (error) {
            console.error("Erreur lors de la reconnaissance de texte avec Tesseract :", error);
            throw error;
        }
    }

    static async getMobileNetPredictions(base64, modelWidth, modelHeight) {
        console.log(`Starting '${Constants.TEXT_DET_TFJS_CONFIG.model.label}' analysis`);

        const canvas = await ImageUtils.createCanvasFromBase64(base64);
        const tmpCanvas = await ImageUtils.createCanvasFromBase64(base64);
        const outputPredictions = [];
        const DET_MEAN = 0.785;
        const DET_STD = 0.275;

        const inputTensor = tf.browser.fromPixels(tmpCanvas);
        const resizedTensor = tf.image.resizeNearestNeighbor(inputTensor, [modelWidth, modelHeight]).toFloat();
        const mean = tf.scalar(255 * DET_MEAN);
        const std = tf.scalar(255 * DET_STD);
        const normalizedTensor = resizedTensor.sub(mean).div(std).expandDims();
        const outputTensor = PredictionUtils.#mobileNetModel.execute(normalizedTensor);
        let prediction = tf.squeeze(outputTensor, 0);
        if (Array.isArray(prediction)) {
            prediction = prediction[0];
        }

        await tf.browser.toPixels(prediction, tmpCanvas);

        // const boundingBoxesWithOpenCV = PredictionUtils.#extractBoundingBoxesWithOpenCV(tmpCanvas, modelWidth, modelHeight);
        const boundingBoxes = PredictionUtils.extractBoundingBoxesWithoutOpenCV(tmpCanvas, modelWidth, modelHeight);

        const ctx = canvas.getContext("2d");
        const scaleX = modelWidth / canvas.width;
        const scaleY = modelHeight / canvas.height;

        for (let i = 0; i < boundingBoxes.length; ++i) {
            const boundingBox = boundingBoxes[i].coordinates;
            const x1 = (modelWidth * boundingBox[0][0]) / scaleX;
            const x2 = (modelWidth * boundingBox[1][0]) / scaleX;
            const y1 = (modelHeight * boundingBox[0][1]) / scaleY;
            const y2 = (modelHeight * boundingBox[3][1]) / scaleY;

            const rect = {
                x: x1,
                y: y1,
                w: x2 - x1,
                h: y2 - y1,
            };

            outputPredictions.push({
                data: ctx.getImageData(rect.x, rect.y, rect.w, rect.h),
                rect: rect,
            });
        }

        tf.dispose([inputTensor, resizedTensor, normalizedTensor, outputTensor]);

        console.log(`${outputPredictions.length} '${Constants.TEXT_DET_TFJS_CONFIG.model.label}' predictions found`);

        return outputPredictions;
    }

    static extractBoundingBoxesWithoutOpenCV(canvas, modelWidth, modelHeight) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        // Convert the image to grayscale
        const grayscaleData = new Uint8Array(modelWidth * modelHeight);
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const gray = 0.299 * r + 0.587 * g + 0.114 * b; // Grayscale conversion formula
            grayscaleData[i / 4] = gray;
        }

        // Apply a binary threshold
        const threshold = 77;
        const binaryData = new Uint8Array(modelWidth * modelHeight);
        for (let i = 0; i < grayscaleData.length; i++) {
            binaryData[i] = grayscaleData[i] > threshold ? 255 : 0;
        }

        // Find contours using a simple edge detection approach
        const contours = [];
        for (let y = 1; y < modelHeight - 1; y++) {
            for (let x = 1; x < modelWidth - 1; x++) {
                const index = y * modelWidth + x;
                if (binaryData[index] === 255) {
                    // Check if the pixel is part of a contour
                    if (
                        binaryData[index - 1] === 0 ||
                        binaryData[index + 1] === 0 ||
                        binaryData[index - modelWidth] === 0 ||
                        binaryData[index + modelWidth] === 0
                    ) {
                        contours.push({ x, y });
                    }
                }
            }
        }

        // Group contours into bounding boxes
        const boundingBoxes = [];
        const visited = new Set();
        for (const contour of contours) {
            if (visited.has(`${contour.x},${contour.y}`)) continue;

            const stack = [contour];
            let minX = contour.x;
            let maxX = contour.x;
            let minY = contour.y;
            let maxY = contour.y;

            while (stack.length > 0) {
                const { x, y } = stack.pop();
                if (x < 0 || x >= modelWidth || y < 0 || y >= modelHeight) continue;
                if (binaryData[y * modelWidth + x] !== 255) continue;
                if (visited.has(`${x},${y}`)) continue;

                visited.add(`${x},${y}`);
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);

                stack.push({ x: x - 1, y });
                stack.push({ x: x + 1, y });
                stack.push({ x, y: y - 1 });
                stack.push({ x, y: y + 1 });
            }

            const width = maxX - minX;
            const height = maxY - minY;
            if (width > 2 && height > 2) {
                const offset = (width * height * 1.8) / (2 * (width + height));
                const p1 = Math.max(0, minX - offset);
                const p2 = Math.min(modelWidth, maxX + offset);
                const p3 = Math.max(0, minY - offset);
                const p4 = Math.min(modelHeight, maxY + offset);

                boundingBoxes.push({
                    id: boundingBoxes.length,
                    config: { stroke: "#ff0000" },
                    coordinates: [
                        [p1 / modelWidth, p3 / modelHeight],
                        [p2 / modelWidth, p3 / modelHeight],
                        [p2 / modelWidth, p4 / modelHeight],
                        [p1 / modelWidth, p4 / modelHeight],
                    ],
                });
            }
        }

        // Reverse the list to fix the inversion
        return boundingBoxes;
    }

    static async getCrnnPredictions(crops, size) {
        function getImageTensorForRecognitionModel(crops, size) {
            const REC_MEAN = 0.694;
            const REC_STD = 0.298;
            return crops.map((imageObject) => {
                let h = imageObject.rect.h;
                let w = imageObject.rect.w;
                let resizeTarget, paddingTarget;
                let aspect_ratio = size[1] / size[0];
                if (aspect_ratio * h > w) {
                    resizeTarget = [size[0], Math.round((size[0] * w) / h)];
                    paddingTarget = [
                        [0, 0],
                        [0, size[1] - Math.round((size[0] * w) / h)],
                        [0, 0],
                    ];
                } else {
                    resizeTarget = [Math.round((size[1] * h) / w), size[1]];
                    paddingTarget = [
                        [0, size[0] - Math.round((size[1] * h) / w)],
                        [0, 0],
                        [0, 0],
                    ];
                }

                const tensor = tf.browser.fromPixels(imageObject.data)
                    .resizeNearestNeighbor(resizeTarget)
                    .pad(paddingTarget, 0)
                    .toFloat()
                    .expandDims();

                const mean = tf.scalar(255 * REC_MEAN);
                const std = tf.scalar(255 * REC_STD);
                const normalized = tensor.sub(mean).div(std);

                tf.dispose([tensor, mean, std]);
                return normalized;
            });
        }

        console.log(`Starting '${Constants.TEXT_REC_TFJS_CONFIG.model.label}' analysis`);

        const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
        const blank = 126;

        const chunkSize = 10;
        const tensors = getImageTensorForRecognitionModel(crops, size);
        const outputPredictions = [];

        for (let i = 0; i < tensors.length; i += chunkSize) {
            const chunk = tensors.slice(i, i + chunkSize);

            const normalizedTensor = tf.concat(chunk);
            chunk.forEach(t => tf.dispose(t));

            const outputTensor = await PredictionUtils.#crnnModel.executeAsync(normalizedTensor);
            const probabilities = tf.softmax(outputTensor, -1);
            const bestPath = tf.unstack(tf.argMax(probabilities, -1), 0);

            for (const sequence of bestPath) {
                let collapsed = "";
                let added = false;
                const values = sequence.dataSync();
                for (const k of values) {
                    if (k === blank) {
                        added = false;
                    } else if (!added) {
                        collapsed += VOCAB[k];
                        added = true;
                    }
                }
                outputPredictions.push(collapsed);
                tf.dispose(sequence);
            }

            tf.dispose([normalizedTensor, outputTensor, probabilities, bestPath]);
        }

        console.log(`${outputPredictions.length} '${Constants.TEXT_REC_TFJS_CONFIG.model.label}' predictions found`);

        return outputPredictions;
    }

    static findPolygonExtremities(overlayData, width, height) {
        // Trouver tous les points rouges
        const redPoints = [];
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const index = (y * width + x) * 4;
                const r = overlayData[index];
                const g = overlayData[index + 1];
                const b = overlayData[index + 2];
                const a = overlayData[index + 3];

                if (r === 255 && g === 0 && b === 0 && a === 150) {
                    redPoints.push({ x, y });
                }
            }
        }

        if (redPoints.length === 0) {
            return null;
        }

        // Trouver le centre de masse
        const center = redPoints.reduce((acc, point) => ({
            x: acc.x + point.x / redPoints.length,
            y: acc.y + point.y / redPoints.length
        }), { x: 0, y: 0 });

        // Calculer la matrice de covariance
        let covXX = 0, covXY = 0, covYY = 0;
        for (const point of redPoints) {
            const dx = point.x - center.x;
            const dy = point.y - center.y;
            covXX += dx * dx;
            covXY += dx * dy;
            covYY += dy * dy;
        }

        // Trouver les vecteurs propres
        const discriminant = Math.sqrt((covXX - covYY) * (covXX - covYY) + 4 * covXY * covXY);
        const lambda1 = (covXX + covYY + discriminant) / 2;
        const lambda2 = (covXX + covYY - discriminant) / 2;

        // Calculer les vecteurs propres
        const eigenvector1 = {
            x: covXY,
            y: lambda1 - covXX
        };
        const eigenvector2 = {
            x: covXY,
            y: lambda2 - covXX
        };

        // Normaliser les vecteurs
        const norm1 = Math.sqrt(eigenvector1.x * eigenvector1.x + eigenvector1.y * eigenvector1.y);
        const norm2 = Math.sqrt(eigenvector2.x * eigenvector2.x + eigenvector2.y * eigenvector2.y);
        eigenvector1.x /= norm1;
        eigenvector1.y /= norm1;
        eigenvector2.x /= norm2;
        eigenvector2.y /= norm2;

        // Projeter tous les points sur les axes principaux
        const projections1 = redPoints.map(point => ({
            point,
            proj: (point.x - center.x) * eigenvector1.x + (point.y - center.y) * eigenvector1.y
        }));
        const projections2 = redPoints.map(point => ({
            point,
            proj: (point.x - center.x) * eigenvector2.x + (point.y - center.y) * eigenvector2.y
        }));

        // Trouver les points extrêmes sur chaque axe
        const min1 = projections1.reduce((min, p) => p.proj < min.proj ? p : min);
        const max1 = projections1.reduce((max, p) => p.proj > max.proj ? p : max);
        const min2 = projections2.reduce((min, p) => p.proj < min.proj ? p : min);
        const max2 = projections2.reduce((max, p) => p.proj > max.proj ? p : max);

        // Créer un ensemble de points candidats
        const candidates = [min1.point, max1.point, min2.point, max2.point];

        // Trouver le point le plus en haut à gauche
        const topLeft = candidates.reduce((min, point) => {
            if (point.y < min.y || (point.y === min.y && point.x < min.x)) {
                return point;
            }
            return min;
        });

        // Trier les points dans le sens horaire en commençant par le point en haut à gauche
        const sortedPoints = candidates.sort((a, b) => {
            if (a === topLeft) return -1;
            if (b === topLeft) return 1;
            
            const angleA = Math.atan2(a.y - topLeft.y, a.x - topLeft.x);
            const angleB = Math.atan2(b.y - topLeft.y, b.x - topLeft.x);
            return angleA - angleB;
        });

        // Ajuster les points pour garantir des coordonnées uniques
        const adjustedPoints = sortedPoints.map((point, index) => {
            const prev = sortedPoints[(index - 1 + 4) % 4];
            const next = sortedPoints[(index + 1) % 4];
            
            let newX = point.x;
            let newY = point.y;
            
            // Calculer la direction du vecteur entre prev et next
            const dx = next.x - prev.x;
            const dy = next.y - prev.y;
            const length = Math.sqrt(dx * dx + dy * dy);
            
            if (length > 0) {
                // Vecteur perpendiculaire
                const perpX = -dy / length;
                const perpY = dx / length;
                
                // Ajuster X si nécessaire
                if (point.x === prev.x || point.x === next.x) {
                    newX = point.x + perpX * 2;
                }
                
                // Ajuster Y si nécessaire
                if (point.y === prev.y || point.y === next.y) {
                    newY = point.y + perpY * 2;
                }
            } else {
                // Si prev et next sont le même point, ajuster dans une direction arbitraire
                newX += 1;
                newY += 1;
            }
            
            return { x: newX, y: newY };
        });

        return adjustedPoints;
    }
}
