
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

const postTool = async (baseUrl, path, body) => {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json; charset=utf-8" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`${path} failed: ${await response.text()}`);
  }
  return await response.json();
};

const runAgent = async (input) => {
  const events = [];
  let state = null;
  const backendUrl = input.pythonToolServerUrl || "http://127.0.0.1:8776";

  const runDeterministicSuperAgent = async () => {
    state = await postTool(backendUrl, "/turn/start", {
      prompt: input.prompt,
      sessionId: input.sessionId,
      memory: input.memory || [],
      lookupCount: input.lookupCount || 0,
    });
    const decision = state?.router_decision;
    if (state.human_confirmation === "rejected" || state.awaiting_human_confirmation || decision?.needs_clarification) {
      state = await postTool(backendUrl, "/agent/finalize", { state });
      return state;
    }
    if (state.human_confirmation === "approved") {
      state = await postTool(backendUrl, "/agent/change", { state });
      state = await postTool(backendUrl, "/agent/finalize", { state });
      return state;
    }
    if (decision?.intent === "meaning_select" || decision?.intent === "da_request") {
      state = await postTool(backendUrl, "/agent/select", { state });
    } else if (decision?.intent === "term_recommend") {
      state = await postTool(backendUrl, "/agent/recommend", { state });
    } else {
      state = await postTool(backendUrl, "/agent/validate", { state });
    }
    state = await postTool(backendUrl, "/agent/finalize", { state });
    return state;
  };

  const startTurnTool = {
    name: "start_turn",
    label: "Start DAsh turn",
    description: "Initialize the DAsh turn, classify intent, and return the mutable AgentTurnState.",
    parameters: Type.Object({
      prompt: Type.String({ description: "The current user request" }),
    }),
    executionMode: "sequential",
    execute: async (_toolCallId, params) => {
      state = await postTool(backendUrl, "/turn/start", {
        prompt: params.prompt || input.prompt,
        sessionId: input.sessionId,
        memory: input.memory || [],
        lookupCount: input.lookupCount || 0,
      });
      return {
        content: [{ type: "text", text: JSON.stringify(state, null, 2) }],
        details: { intent: state?.router_decision?.intent ?? null, human_confirmation: state?.human_confirmation ?? "none" },
      };
    },
  };

  const stateTool = (name, label, path, terminate = false) => ({
    name,
    label,
    description: `${label}. Requires the current AgentTurnState from the previous tool.`,
    parameters: Type.Object({ state: Type.Any({ description: "Current AgentTurnState" }) }),
    executionMode: "sequential",
    execute: async (_toolCallId, params) => {
      state = await postTool(backendUrl, path, { state: params.state || state });
      return {
        content: [{ type: "text", text: JSON.stringify(state, null, 2) }],
        details: { final_answer: state?.final_answer ?? "", lookup_count: state?.lookup_count ?? 0 },
        terminate,
      };
    },
  });

  const model = getModel(input.provider || "openai", input.model || "gpt-4o-mini");
  const agent = new Agent({
    initialState: {
      systemPrompt: [
        "You are the DAsh Super Agent implemented with Pi Agent.",
        "You must orchestrate the workflow by calling tools, not by answering from memory.",
        "Step 1: call start_turn.",
        "If the state needs clarification, is waiting for approval, or has human_confirmation rejected, call finalize_agent.",
        "If intent is meaning_select or da_request, call select_agent and then finalize_agent.",
        "If intent is term_recommend, call recommend_agent and then finalize_agent.",
        "If intent is term_insert, term_update, or term_delete, call validate_agent. If validation is ok and human_confirmation is approved, call change_agent. Then call finalize_agent.",
        "After finalize_agent, answer with state.final_answer exactly.",
      ].join("\n"),
      model,
      thinkingLevel: "low",
      tools: [
        startTurnTool,
        stateTool("select_agent", "Run select_agent", "/agent/select"),
        stateTool("recommend_agent", "Run recommend_agent", "/agent/recommend"),
        stateTool("validate_agent", "Run validate_agent", "/agent/validate"),
        stateTool("change_agent", "Run change_agent", "/agent/change"),
        stateTool("finalize_agent", "Run finalize_agent", "/agent/finalize", true),
      ],
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
  if (!state) {
    events.push({
      type: "tool_execution_start",
      agent: "pi_super_agent",
      message: "deterministic_tool_orchestration",
      payload: { reason: "Pi model did not call tools before idle" },
      timestamp: new Date().toISOString().slice(0, 19) + "+00:00",
    });
    state = await runDeterministicSuperAgent();
    events.push({
      type: "tool_execution_end",
      agent: "pi_super_agent",
      message: "deterministic_tool_orchestration",
      payload: { final_answer: state?.final_answer ?? "" },
      timestamp: new Date().toISOString().slice(0, 19) + "+00:00",
    });
  }
  if (state?.logs) {
    events.push(...state.logs);
  }
  return { events, state };
};

const server = http.createServer(async (request, response) => {
  try {
    if (request.method === "GET" && request.url === "/health") {
      sendJson(response, 200, { ok: true, role: "pi_super_agent", version: "v11" });
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

const port = Number(process.env.PIMONO_SERVER_PORT || "8775");
server.listen(port, "127.0.0.1", () => {
  console.log(`DAsh Pi server listening on http://127.0.0.1:${port}`);
});
