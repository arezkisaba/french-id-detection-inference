/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                framboise: {
                    DEFAULT: "hsl(var(--framboise))",
                    foreground: "hsl(var(--framboise))",
                },
                mangue: {
                    DEFAULT: "hsl(var(--mangue))",
                    foreground: "hsl(var(--mangue))",
                }
            },
        },
    },
    plugins: [],
}

