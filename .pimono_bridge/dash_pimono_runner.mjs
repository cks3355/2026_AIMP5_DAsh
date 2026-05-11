
import { spawn } from "node:child_process";
import process from "node:process";
import { Agent } from "@earendil-works/pi-agent-core";
import { getModel, Type } from "@earendil-works/pi-ai";

if (process.argv.includes("--probe")) {
  process.stdout.write("ok");
  process.exit(0);
}

const readStdin = async () => {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
};

const textFromContent = (content) => {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((block) => block && block.type === "text")
      .map((block) => block.text ?? "")
      .join("");
  }
  return "";
};

const runPythonBackend = (input) =>
  new Promise((resolve, reject) => {
    const child = spawn(input.pythonExecutable, [input.scriptPath, "--pimono-backend"], {
      cwd: process.cwd(),
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => (stdout += chunk.toString("utf8")));
    child.stderr.on("data", (chunk) => (stderr += chunk.toString("utf8")));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || stdout || `Python backend exited with ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`Python backend returned invalid JSON: ${error.message}`));
      }
    });
    child.stdin.end(JSON.stringify(input));
  });

const input = JSON.parse(await readStdin());
const events = [];
let backendState = null;

const dashGraphTool = {
  name: "run_dash_graph",
  label: "Run DAsh graph",
  description: "Route the user request through the DAsh domain graph and return the final state.",
  parameters: Type.Object({
    prompt: Type.String({ description: "The current user request" }),
  }),
  executionMode: "sequential",
  execute: async (_toolCallId, params) => {
    backendState = input.backendState || await runPythonBackend({ ...input, prompt: params.prompt || input.prompt });
    return {
      content: [{ type: "text", text: JSON.stringify(backendState, null, 2) }],
      details: {
        intent: backendState.router_decision?.intent ?? null,
        lookup_count: backendState.lookup_count ?? input.lookupCount,
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
  if (event.type === "tool_execution_end") {
    payload.toolName = event.toolName ?? event.toolCall?.name ?? "run_dash_graph";
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

if (!backendState) {
  backendState = input.backendState || await runPythonBackend(input);
}

process.stdout.write(JSON.stringify({ events, state: backendState }));
