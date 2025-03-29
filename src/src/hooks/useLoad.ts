import { useEffect, useRef } from 'react';

export default function useLoad(effect: () => void, dependencies: any[]) {
    const shouldSkipNextEffect = useRef(false);

    useEffect(() => {
        if (shouldSkipNextEffect.current) {
            shouldSkipNextEffect.current = false;
            return;
        }

        shouldSkipNextEffect.current = true;
        effect();
    }, dependencies);
}