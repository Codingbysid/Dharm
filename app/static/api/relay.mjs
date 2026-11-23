// Runs on Vercel. Only these two routes are needed by the chat client.
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

export default async function handler(req, res) {
  const route = req.query.route;
  const method = { health: 'GET', chat: 'POST' }[route];
  if (!method || method !== req.method) return res.status(405).end();
  const origin = process.env.DHARM_BACKEND_URL;
  const token = process.env.DHARM_BACKEND_TOKEN;
  if (!origin || !token) return res.status(503).json({ detail: 'Inference backend is not configured.' });
  const controller = new AbortController();
  res.on('close', () => controller.abort());
  try {
    const url = new URL('/api/' + route, origin);
    if (url.protocol !== 'https:') throw new Error('Expected HTTPS origin');
    const upstream = await fetch(url, {
      method, redirect: 'error', signal: controller.signal,
      headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/json' },
      body: method === 'POST' ? JSON.stringify(req.body) : undefined
    });
    res.statusCode = upstream.status;
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Content-Type', upstream.headers.get('content-type') || 'application/json');
    if (!upstream.body) return res.end();
    await pipeline(Readable.fromWeb(upstream.body), res);
  } catch {
    if (!res.headersSent && !res.destroyed) return res.status(502).json({ detail: 'Inference service unavailable.' });
    res.end();
  }
}
