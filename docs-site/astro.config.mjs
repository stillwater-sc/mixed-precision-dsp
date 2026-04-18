import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://stillwater-sc.github.io',
  base: '/mixed-precision-dsp',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: 'Mixed-Precision DSP',
      description: 'Header-only C++20 library for mixed-precision digital signal processing',
      customCss: [
        'katex/dist/katex.min.css',
      ],
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/stillwater-sc/mixed-precision-dsp' },
      ],
      editLink: {
        baseUrl: 'https://github.com/stillwater-sc/mixed-precision-dsp/edit/main/docs-site/',
      },
      sidebar: [
        {
          label: 'Getting Started',
          autogenerate: { directory: 'getting-started' },
        },
        {
          label: 'Fundamentals',
          autogenerate: { directory: 'fundamentals' },
        },
        {
          label: 'Signal Conditioning',
          autogenerate: { directory: 'conditioning' },
        },
        {
          label: 'Window Functions',
          autogenerate: { directory: 'windows' },
        },
        {
          label: 'Spectral Analysis',
          autogenerate: { directory: 'spectral' },
        },
        {
          label: 'Image Processing',
          autogenerate: { directory: 'image' },
        },
        {
          label: 'Filter Design',
          autogenerate: { directory: 'filter' },
        },
        {
          label: 'State Estimation',
          autogenerate: { directory: 'estimation' },
        },
        {
          label: 'Mixed-Precision Arithmetic',
          autogenerate: { directory: 'mixed-precision' },
        },
        {
          label: 'API Reference',
          autogenerate: { directory: 'api' },
        },
      ],
    }),
  ],
});
