const addResourcesToCache = async (resources) => {
    const cache = await caches.open("v1");
    await cache.addAll(resources);
};

self.addEventListener("install", (event) => {
    event.waitUntil(
        addResourcesToCache([
            "/favicon.ico"
        ])
    );
});

self.addEventListener('fetch', (event) => {
    //console.log("fetch event =>", event);
});