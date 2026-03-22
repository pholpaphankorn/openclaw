import { describe, it, expect, vi } from "vitest";
import type { OpenClawConfig } from "../config/config.js";
import { createOllamaEmbeddingProvider } from "./embeddings-ollama.js";

describe("embeddings-ollama", () => {
  it("calls /api/embeddings and returns normalized vectors", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ embedding: [3, 4] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { provider } = await createOllamaEmbeddingProvider({
      config: {} as OpenClawConfig,
      provider: "ollama",
      model: "nomic-embed-text",
      fallback: "none",
      remote: { baseUrl: "http://127.0.0.1:11434" },
    });

    const v = await provider.embedQuery("hi");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    // normalized [3,4] => [0.6,0.8]
    expect(v[0]).toBeCloseTo(0.6, 5);
    expect(v[1]).toBeCloseTo(0.8, 5);
  });

  it("resolves baseUrl/apiKey/headers from models.providers.ollama and strips /v1", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ embedding: [1, 0] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { provider } = await createOllamaEmbeddingProvider({
      config: {
        models: {
          providers: {
            ollama: {
              baseUrl: "http://127.0.0.1:11434/v1",
              apiKey: "ollama-\nlocal\r\n", // pragma: allowlist secret
              headers: {
                "X-Provider-Header": "provider",
              },
            },
          },
        },
      } as unknown as OpenClawConfig,
      provider: "ollama",
      model: "",
      fallback: "none",
    });

    await provider.embedQuery("hello");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:11434/api/embeddings",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json",
          Authorization: "Bearer ollama-local",
          "X-Provider-Header": "provider",
        }),
      }),
    );
  });

  it("fails fast when memory-search remote apiKey is an unresolved SecretRef", async () => {
    await expect(
      createOllamaEmbeddingProvider({
        config: {} as OpenClawConfig,
        provider: "ollama",
        model: "nomic-embed-text",
        fallback: "none",
        remote: {
          baseUrl: "http://127.0.0.1:11434",
          apiKey: { source: "env", provider: "default", id: "OLLAMA_API_KEY" },
        },
      }),
    ).rejects.toThrow(/agents\.\*\.memorySearch\.remote\.apiKey: unresolved SecretRef/i);
  });

  it("falls back to env key when models.providers.ollama.apiKey is an unresolved SecretRef", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ embedding: [1, 0] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    vi.stubEnv("OLLAMA_API_KEY", "ollama-env");

    const { provider } = await createOllamaEmbeddingProvider({
      config: {
        models: {
          providers: {
            ollama: {
              baseUrl: "http://127.0.0.1:11434/v1",
              apiKey: { source: "env", provider: "default", id: "OLLAMA_API_KEY" },
              models: [],
            },
          },
        },
      } as unknown as OpenClawConfig,
      provider: "ollama",
      model: "nomic-embed-text",
      fallback: "none",
    });

    await provider.embedQuery("hello");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:11434/api/embeddings",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer ollama-env",
        }),
      }),
    );
  });

  it("calls /api/embed for embedBatch with batch texts", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            model: "nomic-embed-text",
            embeddings: [
              [3, 4],
              [5, 12],
            ],
          }),
          {
            status: 200,
            headers: { "content-type": "application/json" },
          },
        ),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const { provider } = await createOllamaEmbeddingProvider({
      config: {} as OpenClawConfig,
      provider: "ollama",
      model: "nomic-embed-text",
      fallback: "none",
      remote: { baseUrl: "http://127.0.0.1:11434" },
    });

    const vectors = await provider.embedBatch(["hi", "hello"]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:11434/api/embed",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ model: "nomic-embed-text", input: ["hi", "hello"] }),
      }),
    );
    // normalized [3,4] => [0.6,0.8], [5,12] => ~[0.38,0.92]
    expect(vectors[0][0]).toBeCloseTo(0.6, 5);
    expect(vectors[0][1]).toBeCloseTo(0.8, 5);
    expect(vectors[1][0]).toBeCloseTo(0.384615, 5);
    expect(vectors[1][1]).toBeCloseTo(0.923077, 5);
  });

  it("falls back to sequential /api/embeddings if /api/embed fails", async () => {
    const fetchMocks = {
      embed: vi.fn(
        async () =>
          new Response(JSON.stringify({ error: "not found" }), {
            status: 404,
            headers: { "content-type": "application/json" },
          }),
      ),
      embeddings: vi.fn(
        async () =>
          new Response(JSON.stringify({ embedding: [3, 4] }), {
            status: 200,
            headers: { "content-type": "application/json" },
          }),
      ),
    };
    globalThis.fetch = ((url: RequestInfo) => {
      const urlStr = url instanceof Request ? url.url : String(url);
      if (urlStr.includes("/api/embed") && !urlStr.includes("/api/embeddings")) {
        return fetchMocks.embed();
      }
      return fetchMocks.embeddings();
    }) as unknown as typeof fetch;

    const { provider } = await createOllamaEmbeddingProvider({
      config: {} as OpenClawConfig,
      provider: "ollama",
      model: "nomic-embed-text",
      fallback: "none",
      remote: { baseUrl: "http://127.0.0.1:11434" },
    });

    const vectors = await provider.embedBatch(["hi", "hello"]);
    // Should have made 2 requests to /api/embeddings (one per text)
    expect(fetchMocks.embeddings).toHaveBeenCalledTimes(2);
    // Should have tried /api/embed first
    expect(fetchMocks.embed).toHaveBeenCalledTimes(1);
    // Vectors should be normalized [3,4] => [0.6,0.8]
    expect(vectors[0][0]).toBeCloseTo(0.6, 5);
    expect(vectors[0][1]).toBeCloseTo(0.8, 5);
  });
});
