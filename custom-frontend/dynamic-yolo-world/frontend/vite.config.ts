import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
 import {version} from './package.json';

// https://vite.dev/config/
export default defineConfig({
	base: `/${version}`,
	plugins: [react(),],
	// This is needed by FoxGlove
	define: {
		global: {},
	},
	worker: {
		format: "es",
	},
	build: {
		rollupOptions: {
			output: {
				format: "esm",
			},
		},
	},
});
