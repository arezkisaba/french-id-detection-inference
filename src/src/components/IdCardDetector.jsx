import React, { useRef, useEffect, useState } from "react";
import useLoad from "../hooks/useLoad";
import RealTimeCapture from "./RealTimeCapture.jsx";
import { ImageUtils } from "../utils/ImageUtils.js";
import PredictionUtils from '../utils/PredictionUtils.js';
import Loader from "./Loader.jsx";

const IdCardDetector = () => {
    const [isLoadingModels, setIsLoadingModels] = useState(false);
    const [status, setStatus] = useState({ realTimeCaptureActive: false, lastBase64: undefined });

    useLoad(() => {
        setIsLoadingModels(true);
        PredictionUtils.loadModels().then(() => {
            setIsLoadingModels(false);
            console.log("All models loaded successfully");
        });
        return () => { };
    }, []);

    useEffect(() => {
        if (!status.realTimeCaptureActive && status.lastBase64) {
            handleFileSelection(status.lastBase64);
        }
    }, [status]);

    return (
        <>{isLoadingModels ? <Loader /> : getContent(status.realTimeCaptureActive)}</>
    );

    function getContent(isCameraOpen) {
        if (isCameraOpen) {
            return (
                <div>
                    <RealTimeCapture handleFileSelection={(base64) => onRealTimeCaptureEnded(base64)} />
                </div>
            );
        }

        return (
            <div className="flex flex-col gap-4 p-4">
                <div className="flex flex-row items-center justify-start gap-4">
                    <button
                        onClick={startRealTimeCapture}
                        className="px-6 py-2 text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors w-[220px]">
                        Real Time Capture
                    </button>
                    <button
                        onClick={triggerFileSelection}
                        className="px-6 py-2 text-white bg-green-600 rounded-lg hover:bg-green-700 transition-colors w-[220px]">
                        Image Selection
                    </button>
                    <button
                        onClick={triggerOCR}
                        className="px-6 py-2 text-white bg-red-600 rounded-lg hover:bg-red-700 transition-colors w-[220px]">
                        OCR
                    </button>
                </div>
                <canvas id="canvas" className="w-full border border-gray-300 rounded-lg" />
                <canvas id="canvasCropped" className="w-full border border-gray-300 rounded-lg" />
                <canvas id="canvasSegmented" className="w-full border border-gray-300 rounded-lg" />
            </div>
        );
    }

    function startRealTimeCapture() {
        setStatus({ realTimeCaptureActive: true, lastBase64: undefined });
    }

    function onRealTimeCaptureEnded(base64) {
        setStatus({ realTimeCaptureActive: false, lastBase64: base64 });
    }

    function triggerFileSelection() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = (e) => {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (e) => {
                const base64 = e.target.result;
                handleFileSelection(base64);
            };
            reader.readAsDataURL(file);
        };
        input.click();
    }

    async function triggerOCR() {
        const canvas = document.getElementById('canvasSegmented');
        const base64 = canvas.toDataURL();
        const predictions = await PredictionUtils.getOcrText(base64);
    }

    async function handleFileSelection(base64) {
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = async () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            const prediction = await PredictionUtils.getBestDocumentPrediction(base64, undefined, img.width, img.height, true);
            if (!prediction) {
                console.log("No prediction found");
                return;
            }

            const canvasCropped = document.getElementById('canvasCropped');
            canvasCropped.width = prediction.rect.w;
            canvasCropped.height = prediction.rect.h;
            const ctxCropped = canvasCropped.getContext('2d');
            if (prediction) {
                const imageData = ctx.getImageData(prediction.rect.x, prediction.rect.y, prediction.rect.w, prediction.rect.h);
                ctxCropped.putImageData(imageData, 0, 0);
                drawPrediction(ctx, prediction);
                // drawSegmentationMask(ctx, prediction);
                // const canvasSegmented = document.getElementById('canvasSegmented');
                // await ImageUtils.cropImageWithAngleCorrection(base64, prediction.seg, canvasSegmented);
            }
        };
        img.src = base64;
    }

    function drawPrediction(ctx, prediction) {
        ctx.lineWidth = 10;
        ctx.strokeStyle = "black";
        ctx.strokeRect(
            prediction.rect.x,
            prediction.rect.y,
            prediction.rect.w,
            prediction.rect.h
        );
        if (prediction.className) {
            ctx.font = "35px Arial";
            ctx.fillStyle = "black";
            const textX = prediction.rect.x;
            const textY = prediction.rect.y - 20;
            ctx.fillText(`${prediction.className} ${prediction.probability}%`, textX, textY);
        }
    }

    function drawSegmentationMask(ctx, prediction) {
        ctx.lineWidth = 10;
        if (prediction.seg) {
            prediction.seg.possiblePolygons.forEach(polygon => {
                drawPolygon(ctx, polygon.points, polygon.intersectScore);
            });
        }
    }

    function drawPolygon(ctx, coordinates, intersectScore) {
        ctx.beginPath();
        ctx.moveTo(coordinates[0].x, coordinates[0].y);
        coordinates.forEach(point => {
            ctx.lineTo(point.x, point.y);
        });
        ctx.closePath();
        ctx.lineWidth = 10;
        const colors = ["red", "green", "blue", "yellow", "orange", "pink", "purple", "cyan", "lime", "violet", "gold", "brown"];
        const selectedColorIndex = Math.floor(Math.random() * colors.length);
        const selectedColor = colors[selectedColorIndex];
        colors.splice(selectedColorIndex, 1);
        ctx.strokeStyle = selectedColor;
        ctx.stroke();

        console.log(selectedColor, intersectScore, coordinates);
    }
};

export default IdCardDetector;