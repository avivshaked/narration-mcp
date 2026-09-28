// A Node.js MCP-client stand-in for spike (k): start the server with child_process.spawn (libuv's uv_spawn,
// which assigns the child to libuv's global Job Object), wait until the server has reported, write what
// this client saw, and exit at once. Node's exit closes the job's handle; KILL_ON_JOB_CLOSE then ends
// whatever is still in it.
//
//   node node_client.js <server_out> <client_out> <command> <args...>
"use strict";
const { spawn } = require("node:child_process");
const fs = require("node:fs");

const [serverOut, clientOut, command, ...args] = process.argv.slice(2);
const started = Date.now();
const child = spawn(command, args, { stdio: ["pipe", "pipe", "inherit"] });
child.stdout.on("data", () => {});
child.on("error", (error) => {
  fs.writeFileSync(clientOut, JSON.stringify({ node: process.version, spawn_error: String(error) }));
  process.exit(4);
});

const timer = setInterval(() => {
  if (fs.existsSync(serverOut)) {
    clearInterval(timer);
    fs.writeFileSync(
      clientOut,
      JSON.stringify({ node: process.version, spawned_server_pid: child.pid, server_reported: true, waited_ms: Date.now() - started }),
    );
    process.exit(0); // the session ends; libuv's job closes with this process
  }
  if (Date.now() - started > 60000) {
    clearInterval(timer);
    fs.writeFileSync(clientOut, JSON.stringify({ node: process.version, spawned_server_pid: child.pid, server_reported: false }));
    child.kill();
    process.exit(3);
  }
}, 50);
