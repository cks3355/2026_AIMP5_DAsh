
import http from "node:http";
import process from "node:process";
import { Agent } from "@earendil-works/pi-agent-core";
import { getModel, Type } from "@earendil-works/pi-ai";

const readRequest = async (request) => {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
};

const sendJson = (response, status, body) => {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(body));
};

const runAgent = async (input) => {
  const events = [];
  let backendState = null;

  const dashGraphTool = {
    name: "run_dash_graph",
    label: "Run DAsh graph",
    description: "Return the precomputed DAsh routing, lookup, validation, and final answer state.",
    parameters: Type.Object({
      prompt: Type.String({ description: "The current user request" }),
    }),
    executionMode: "sequential",
    execute: async () => {
      backendState = input.backendState;
      return {
        content: [{ type: "text", text: JSON.stringify(backendState, null, 2) }],
        details: {
          intent: backendState?.router_decision?.intent ?? null,
          lookup_count: backendState?.lookup_count ?? input.lookupCount,
        },
        terminate: true,
      };
    },
  };

  const model = getModel(input.provider || "openai", input.model || "gpt-4o-mini");
  const agent = new Agent({
    initialState: {
      systemPrompt: [
        "You are the DAsh Super Agent.",
        "For every user request, call run_dash_graph exactly once.",
        "Do not answer from memory before the tool result is available.",
        "The tool result is the authoritative DAsh routing, validation, lookup, and final answer state.",
      ].join("\n"),
      model,
      thinkingLevel: "low",
      tools: [dashGraphTool],
      messages: input.memory || [],
    },
    convertToLlm: (messages) =>
      messages.filter((m) => ["user", "assistant", "toolResult"].includes(m.role)),
  });

  agent.subscribe((event) => {
    const payload = { pi_event_type: event.type };
    if (event.type === "message_update" && event.assistantMessageEvent?.type === "text_delta") {
      payload.delta = event.assistantMessageEvent.delta;
    }
    if (event.type === "tool_execution_start" || event.type === "tool_execution_end") {
      payload.toolName = event.toolName ?? "run_dash_graph";
    }
    events.push({
      type: event.type,
      agent: "pimono_agent",
      message: payload.delta || payload.toolName || event.type,
      payload,
      timestamp: new Date().toISOString().slice(0, 19) + "+00:00",
    });
  });

  await agent.prompt(input.prompt);
  await agent.waitForIdle();
  return { events, state: backendState || input.backendState };
};

const server = http.createServer(async (request, response) => {
  try {
    if (request.method === "GET" && request.url === "/health") {
      sendJson(response, 200, { ok: true });
      return;
    }
    if (request.method === "POST" && request.url === "/run") {
      const input = JSON.parse(await readRequest(request));
      sendJson(response, 200, await runAgent(input));
      return;
    }
    sendJson(response, 404, { error: "not found" });
  } catch (error) {
    sendJson(response, 500, { error: error.message, stack: error.stack });
  }
});

const port = Number(process.env.PIMONO_SERVER_PORT || "8765");
server.listen(port, "127.0.0.1", () => {
  console.log(`DAsh Pi server listening on http://127.0.0.1:${port}`);
});
