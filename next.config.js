/** @type {import('next').NextConfig} */
const withBundleAnalyzer = require("@next/bundle-analyzer")({
    enabled: process.env.ANALYZE === "true",
});

// GitHub Pages project-page base path (repo name). Applied only to production
// builds so `next dev` still serves at the root.
const basePath = process.env.NODE_ENV === "production" ? "/llm-inference-viz" : "";

const nextConfig = {
    output: "export",              // static export for GitHub Pages
    basePath,
    env: { BASE_URL: basePath },   // inlined for runtime asset fetches (wasm/json/img)
    trailingSlash: true,
    images: { unoptimized: true }, // no image-optimization server on Pages
    reactStrictMode: false,
    productionBrowserSourceMaps: true,
    experimental: {
        appDir: true,
    },
};

module.exports = withBundleAnalyzer(nextConfig);
