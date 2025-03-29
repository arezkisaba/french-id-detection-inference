import { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as Constants from "../common/constants.js";
import { ImageUtils } from '../utils/ImageUtils.js';
import PredictionUtils from '../utils/PredictionUtils';

const RealTimeCapture = ({ handleFileSelection }) => {
    const motionThreshold = 0.08;
    const videoRef = useRef(null);
    const tempCanvasRef = useRef(null);
    const tempCanvasCtxRef = useRef(null);
    const maskCanvasRef = useRef(null);
    const maskCanvasCtxRef = useRef(null);
    let lastPrediction = null;
    const previousFrameRef = useRef(null);
    const frameIndexRef = useRef(0);
    const streamRef = useRef(null);

    useEffect(() => {
        startCamera();
        return () => {
            stopCamera();
        };
    }, []);

    return (
        <>
            <div style={{ display: "flex", flexDirection: "column", width: "100%" }}>
                <div style={{ position: "relative", width: "100%" }}>
                    <video
                        ref={videoRef}
                        style={{ display: "block", zIndex: 2, }}
                        autoPlay
                        playsInline
                        muted
                    />
                    <canvas
                        ref={maskCanvasRef}
                        style={{ position: "absolute", top: 0, left: 0, zIndex: 3, pointerEvents: "none" }}
                    />
                </div>
                <canvas
                    ref={tempCanvasRef}
                    style={{
                        width: '100%',
                        zIndex: 2,
                        pointerEvents: "none"
                    }}
                />
            </div>
        </>
    );

    async function startCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "environment",
                width: { ideal: 1920 },
                height: { ideal: 1080 },
                advanced: [{ focusMode: 'continuous' }]
            }
        });
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
            console.log('Camera started');
            detectionLoop();
        };
    };

    async function stopCamera() {
        if (streamRef.current) {
            const tracks = streamRef.current.getTracks();
            tracks.forEach(track => {
                track.stop();
                streamRef.current.removeTrack(track);
            });
            streamRef.current = null;
            if (videoRef.current) {
                videoRef.current.srcObject = null;
                videoRef.current.load();
            }
            console.log('Camera stopped');
        }
    };

    async function detectionLoop() {
        initializeUiComponents(Constants.DOC_SEG_TFJS_CONFIG.model.width, Constants.DOC_SEG_TFJS_CONFIG.model.height);

        const captureFrame = async () => {

            if (videoRef.current == null) {
                return;
            }

            tf.engine().startScope();
            let predictionToApply = null;

            let isFrameStabilized = false;
            if (frameIndexRef.current === 0) {
                const currentFrame = ImageUtils.convertVideoToImageData(videoRef.current, videoRef.current.videoWidth, videoRef.current.videoHeight);
                if (previousFrameRef.current) {
                    const motionPercentage = ImageUtils.calculateMotionPercentage(currentFrame, previousFrameRef.current);
                    isFrameStabilized = motionPercentage < motionThreshold;
                    console.log(`isFrameStabilized : ${isFrameStabilized} (moved by ${(motionPercentage * 100).toFixed(2)}%)`);
                }

                previousFrameRef.current = currentFrame;
            }

            const base64 = undefined;//ImageUtils.convertVideoFrameToBase64(videoRef.current);
            const newPrediction = await PredictionUtils.getBestDocumentPrediction(
                base64,
                videoRef.current,
                videoRef.current.videoWidth,
                videoRef.current.videoHeight,
                false
            );
            if (newPrediction) {
                if (!lastPrediction) {
                    predictionToApply = newPrediction;
                } else {
                    predictionToApply = lastPrediction;
                    if (isFrameStabilized) {
                        predictionToApply = newPrediction;
                        const base64Image = ImageUtils.convertVideoFrameToBase64(videoRef.current);
                        const img = new Image();
                        img.src = base64Image;
                        img.onload = () => {
                            const canvas = document.createElement('canvas');
                            const ctx = canvas.getContext('2d');
                            canvas.width = img.width;
                            canvas.height = img.height;
                            ctx.drawImage(img, 0, 0);
                            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                            const sharpness = calculateImageSharpness(imageData);
                            if (sharpness > 0.05) {
                                handleFileSelection(base64Image);
                            }
                        };
                    } else {
                        predictionToApply = newPrediction;
                    }
                }

                lastPrediction = predictionToApply;
            }

            if (maskCanvasRef.current == null) {
                return;
            }

            maskCanvasCtxRef.current.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
            if (predictionToApply) {
                drawPrediction(maskCanvasCtxRef, predictionToApply);
            }

            frameIndexRef.current++;
            if (frameIndexRef.current > 5) {
                frameIndexRef.current = 0;
            }

            tf.engine().endScope();
            requestAnimationFrame(captureFrame);
        };

        requestAnimationFrame(captureFrame);
    };

    function calculateImageSharpness(imageData) {

        // Look at laplacian of gaussian

        const width = imageData.width;
        const height = imageData.height;
        const data = imageData.data;
        let sum = 0;
        let count = 0;

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = (y * width + x) * 4;
                
                // Convert to grayscale using luminance formula
                const gray = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
                
                // Horizontal gradient
                const gx = 
                    -1 * (data[((y-1) * width + (x-1)) * 4]) +
                    -2 * (data[((y) * width + (x-1)) * 4]) +
                    -1 * (data[((y+1) * width + (x-1)) * 4]) +
                    1 * (data[((y-1) * width + (x+1)) * 4]) +
                    2 * (data[((y) * width + (x+1)) * 4]) +
                    1 * (data[((y+1) * width + (x+1)) * 4]);

                // Vertical gradient
                const gy = 
                    -1 * (data[((y-1) * width + (x-1)) * 4]) +
                    -2 * (data[((y-1) * width + (x)) * 4]) +
                    -1 * (data[((y-1) * width + (x+1)) * 4]) +
                    1 * (data[((y+1) * width + (x-1)) * 4]) +
                    2 * (data[((y+1) * width + (x)) * 4]) +
                    1 * (data[((y+1) * width + (x+1)) * 4]);

                const gradient = Math.sqrt(gx * gx + gy * gy);
                sum += gradient;
                count++;
            }
        }

        const sharpness = sum / count / 255; // Normalize by 255 to get values between 0-1
        console.log('Image sharpness:', sharpness);
        return sharpness;
    }

    function initializeUiComponents(modelWidth, modelHeight) {
        const modelRatioX = 1;// modelWidth / videoRef.current.videoWidth;
        const modelRatioY = 1;//modelHeight / videoRef.current.videoHeight;
        const screenWidth = window.innerWidth;
        const displayRatio = screenWidth / videoRef.current.videoWidth;
        const displayWidth = screenWidth;
        const displayHeight = videoRef.current.videoHeight * displayRatio;
        videoRef.current.style.width = `${Math.round(displayWidth)}px`;
        videoRef.current.style.height = `${Math.round(displayHeight)}px`;
        maskCanvasRef.current.width = videoRef.current.videoWidth * modelRatioX;
        maskCanvasRef.current.height = videoRef.current.videoHeight * modelRatioY;
        maskCanvasRef.current.style.width = `${Math.round(displayWidth)}px`;
        maskCanvasRef.current.style.height = `${Math.round(displayHeight)}px`;
        maskCanvasCtxRef.current = maskCanvasRef.current.getContext("2d", { willReadFrequently: true });
        tempCanvasRef.current.width = videoRef.current.videoWidth;
        tempCanvasRef.current.height = videoRef.current.videoHeight;
        tempCanvasRef.current.style.width = `${Math.round(displayWidth)}px`;
        tempCanvasRef.current.style.height = `${Math.round(displayHeight)}px`;
        tempCanvasCtxRef.current = tempCanvasRef.current.getContext("2d");
    }

    function drawPrediction(maskCanvasCtxRef, predictionToApply) {
        maskCanvasCtxRef.current.fillStyle = 'rgba(0, 0, 0, 0.8)';
        maskCanvasCtxRef.current.clearRect(
            predictionToApply.rect.x,
            predictionToApply.rect.y,
            predictionToApply.rect.w,
            predictionToApply.rect.h
        );
        if (predictionToApply.maskImg) {
            maskCanvasCtxRef.current.putImageData(predictionToApply.maskImg, 0, 0);
        }
        maskCanvasCtxRef.current.lineWidth = 10;
        maskCanvasCtxRef.current.strokeStyle = "white";
        maskCanvasCtxRef.current.strokeRect(
            predictionToApply.rect.x,
            predictionToApply.rect.y,
            predictionToApply.rect.w,
            predictionToApply.rect.h
        );
        if (predictionToApply.className) {
            maskCanvasCtxRef.current.font = "35px Arial";
            maskCanvasCtxRef.current.fillStyle = "white";
            const textX = predictionToApply.rect.x;
            const textY = predictionToApply.rect.y - 20;
            maskCanvasCtxRef.current.fillText(`${predictionToApply.className} ${predictionToApply.probability}%`, textX, textY);
        }
    }
};

export default RealTimeCapture;