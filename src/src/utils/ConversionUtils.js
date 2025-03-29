import { md5 } from "js-md5";

export class ConversionUtils {

    static generateHash(toEncode) {
        return md5.base64(toEncode);
    }

    static convertFromBinaryToBase64(toEncode) {
        const uint8Array = new Uint8Array(toEncode);
        const data = uint8Array.reduce((acc, i) => acc += String.fromCharCode.apply(null, [i]), '');

        return btoa(data);
    }

    static convertFromBase64ToBinary(toEncode) {
        let data = atob(toEncode);
        let result = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) {
            result[i] = data.charCodeAt(i);
        }

        return result;
    }
}
