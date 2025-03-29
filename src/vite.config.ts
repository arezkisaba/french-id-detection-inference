import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';

export default defineConfig({
    plugins: [react()],
    optimizeDeps: {
        exclude: ['onnxruntime-web']
    },
    worker: {
        format: 'es'
    },
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url))
        }
    },
    server: {
        https: {
            key: fs.readFileSync('../certs/192.168.1.100-key.pem'),
            cert: fs.readFileSync('../certs/192.168.1.100.pem')
        },
        host: '0.0.0.0',
        port: 12345,
        hmr: {
            host: '192.168.1.100',
            port: 12345
        },
        headers: {
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp'
        }
    }
})
