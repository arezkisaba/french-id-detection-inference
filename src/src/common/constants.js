export const DOC_SEG_ONNX_CONFIG = {
    model: {
        label: "yolo (seg)",
        batchSize: 1,
        channels: 3,
        height: 320,
        width: 320,
        path: "/models/yolo/best.onnx",
    },
};

export const DOC_SEG_TFJS_CONFIG = {
    model: {
        label: "yolo (seg)",
        batchSize: 1,
        channels: 3,
        height: 320,
        width: 320,
        path: "/models/best_web_model_quantized/model.json"
    },
};

export const TEXT_DET_TFJS_CONFIG = {
    model: {
        label: "db_mobilenet_v2",
        height: 512,
        width: 512,
        path: "/models/db_mobilenet_v2/model.json",
    },
};

export const TEXT_REC_TFJS_CONFIG = {
    model: {
        label: "crnn_mobilenet_v2",
        height: 32,
        width: 128,
        path: "/models/crnn_mobilenet_v2/model.json",
    },
};