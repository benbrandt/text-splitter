import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'text-splitter',
			social: {
				github: 'https://github.com/benbrandt/text-splitter',
			},
			sidebar: [
				{
					label: 'Guides',
					autogenerate: { directory: 'guides' },
					// items: [
					// 	// Each item here is one entry in the navigation menu.
					// 	{ label: 'Example Guide', link: '/guides/example/' },
					// ],
				},
				{
					label: 'Reference',
					items: [
						// Each item here is one entry in the navigation menu.
						{ label: 'Rust API', link: 'https://docs.rs/text-splitter' },
						{ label: 'Python API', link: 'https://semantic-text-splitter.readthedocs.io/en/latest/semantic_text_splitter.html' },
					],
				},
			],
		}),
	],
});
