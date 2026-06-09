// ════════════════════════════════════════════════════════════════
//  HADO Smart Court IoT — MacBook Backend Server
//  • Receives 'hado_stream' from Pi (player positions + vest hue)
//  • Maintains 40-second sliding window buffer
//  • Hybrid role classifier (color: 0–30 s, behavior: 30 s+)
//  • Runs tactical analysis every 2 seconds
//  • Broadcasts 'tactical_advice' and 'player_roles'
// ════════════════════════════════════════════════════════════════

const express = require('express');
const http    = require('http');
const { Server } = require('socket.io');

const app    = express();
const server = http.createServer(app);
const io     = new Server(server, { cors: { origin: "*" } });

// ───── Configuration ──────────────────────────────────────────
const PORT             = 8080;
const COURT_CENTER_X   = 320.0;
const COURT_MID_Y      = 240.0;
const BUFFER_WINDOW    = 40_000;   // 40-second sliding window
const ANALYSIS_PERIOD  = 2_000;    // tactical analysis every 2 s
const ROLE_WINDOW      = 30_000;   // 30-second role behavior window
const COLD_START_MS    = 30_000;   // first 30 s → color-dominant
const ATTACK_THRESH_PX = 30;       // wrist above shoulder by 30 px

// ───── Color → Role table (HSV hue 0–179) ─────────────────────
// Configure per your vest colors. Each role has one or more hue ranges
// because red wraps around 0/180.
const HUE_ROLE_TABLE = [
  { role: "Attacker",   ranges: [[0, 10], [170, 180]] },  // red
  { role: "Technician", ranges: [[20, 40]] },              // yellow
  { role: "Defender",   ranges: [[90, 130]] },             // blue
];

function hueToRole(hue) {
  if (hue == null || hue < 0) return null;
  for (const entry of HUE_ROLE_TABLE) {
    for (const [lo, hi] of entry.ranges) {
      if (hue >= lo && hue <= hi) return entry.role;
    }
  }
  return null;
}

// ───── State ──────────────────────────────────────────────────
let dataBuffer = [];
const SERVER_START = Date.now();

// ───── Helpers ────────────────────────────────────────────────
function stdDev(arr) {
  if (arr.length < 2) return 0;
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

function mostCommon(arr) {
  if (arr.length === 0) return null;
  const counts = {};
  for (const v of arr) counts[v] = (counts[v] || 0) + 1;
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

// ════════════════════════════════════════════════════════════════
//  HYBRID ROLE CLASSIFIER
// ════════════════════════════════════════════════════════════════
//  Stage 1 (0–30 s)   : vest color is the primary signal
//  Stage 2 (30 s +)   : behavior overrides; color used as tiebreak
//  Output: { playerId: { team, role, confidence, source } }
// ════════════════════════════════════════════════════════════════
function classifyRoles(buffer) {
  const now      = Date.now();
  const elapsed  = now - SERVER_START;
  const coldStart = elapsed < COLD_START_MS;

  // 1) Group recent samples by player
  const players = {};
  for (const d of buffer) {
    if (d.timestamp < now - ROLE_WINDOW) continue;
    if (!players[d.playerId]) {
      players[d.playerId] = {
        team: d.x < COURT_CENTER_X ? "A" : "B",
        samples: [], xs: [], ys: [],
        attackCount: 0, forwardCount: 0,
        shieldCount: 0, chargeCount: 0,
        hueVotes: [],
      };
    }
    const p = players[d.playerId];
    p.samples.push(d);
    p.xs.push(d.x);
    p.ys.push(d.y);

    // Attack motion: prefer Pi-side pose label, fallback to wrist-above-shoulder
    if (d.pose === "attack") {
      p.attackCount++;
    } else if (d.pose == null &&
               d.rightShoulderY - d.rightWristY > ATTACK_THRESH_PX) {
      p.attackCount++;
    }
    if (d.pose === "shield") p.shieldCount = (p.shieldCount || 0) + 1;
    if (d.pose === "charge") p.chargeCount = (p.chargeCount || 0) + 1;

    // Forward-zone occupancy
    const inEnemyZone = (p.team === "A" && d.x > COURT_CENTER_X) ||
                        (p.team === "B" && d.x < COURT_CENTER_X);
    if (inEnemyZone) p.forwardCount++;

    // Vest color vote
    const colorRole = hueToRole(d.vestHue);
    if (colorRole) p.hueVotes.push(colorRole);
  }

  // 2) Compute behavior score & color majority per player
  const stats = Object.entries(players).map(([pid, p]) => {
    const n = p.samples.length || 1;
    const attackRatio  = p.attackCount  / n;
    const forwardRatio = p.forwardCount / n;
    const mobility     = stdDev(p.xs) + stdDev(p.ys);
    const colorRole    = mostCommon(p.hueVotes);
    return {
      pid:          Number(pid),
      team:         p.team,
      attackScore:  attackRatio * 2 + forwardRatio,
      mobility,
      colorRole,
      colorVotes:   p.hueVotes.length,
      sampleCount:  n,
    };
  });

  // 3) Build final role map
  const roles = {};
  for (const team of ["A", "B"]) {
    const teamPlayers = stats.filter(s => s.team === team);
    if (teamPlayers.length === 0) continue;
    // Confidence penalty when fewer than 3 players detected on a team
    const teamSizePenalty = teamPlayers.length >= 3 ? 1.0
                          : teamPlayers.length === 2 ? 0.7
                          : 0.4;

    if (coldStart) {
      // STAGE 1 — Color-dominant
      // Take each player's most-voted color role; fill gaps if any.
      const taken = new Set();
      for (const p of teamPlayers) {
        if (p.colorRole && !taken.has(p.colorRole)) {
          roles[p.pid] = {
            team:       p.team,
            role:       p.colorRole,
            confidence: Math.min(1, p.colorVotes / 5) * teamSizePenalty,
            source:     "color",
          };
          taken.add(p.colorRole);
        }
      }
      // Players without a color match: behavior fallback
      const unranked = teamPlayers.filter(p => !roles[p.pid]);
      unranked.sort((a, b) =>
        b.attackScore - a.attackScore || b.mobility - a.mobility
      );
      const remaining = ["Attacker", "Technician", "Defender"]
        .filter(r => !taken.has(r));
      unranked.forEach((p, i) => {
        if (i < remaining.length) {
          roles[p.pid] = {
            team:       p.team,
            role:       remaining[i],
            confidence: 0.3 * teamSizePenalty,
            source:     "behavior-fallback",
          };
        }
      });
    } else {
      // STAGE 2 — Behavior-dominant
      // Rank within team by attack score (mobility tiebreak).
      teamPlayers.sort((a, b) =>
        b.attackScore - a.attackScore || b.mobility - a.mobility
      );
      const order = ["Attacker", "Technician", "Defender"];
      teamPlayers.forEach((p, i) => {
        if (i >= 3) return;
        const behaviorRole = order[i];
        const agree = p.colorRole === behaviorRole;
        roles[p.pid] = {
          team:       p.team,
          role:       behaviorRole,
          confidence: (agree ? 0.95 : 0.65) * teamSizePenalty,
          source:     agree ? "behavior+color" : "behavior",
        };
      });
    }
  }

  return { roles, coldStart, elapsed };
}

// ════════════════════════════════════════════════════════════════
//  Tactical Analysis (every 2 seconds) — degrades gracefully
//  Works with as few as 1 player detected on either side.
// ════════════════════════════════════════════════════════════════
function uniqueByPlayer(samples) {
  // Keep only the most recent sample per playerId
  const latest = new Map();
  for (const d of samples) {
    const prev = latest.get(d.playerId);
    if (!prev || d.timestamp > prev.timestamp) latest.set(d.playerId, d);
  }
  return [...latest.values()];
}

function analyzeTactics(buffer) {
  const ourTeam   = uniqueByPlayer(buffer.filter(d => d.x < COURT_CENTER_X));
  const enemyTeam = uniqueByPlayer(buffer.filter(d => d.x >= COURT_CENTER_X));
  const ourCount   = ourTeam.length;
  const enemyCount = enemyTeam.length;

  // No one in frame at all
  if (ourCount + enemyCount === 0) {
    return { msg: "👁️ Waiting for players to appear...", mode: "idle" };
  }

  // ── Single-side visibility (only one team detected) ──
  if (enemyCount === 0) {
    // Only our team visible — give positional feedback
    const ourAvgX = ourTeam.reduce((a, d) => a + d.x, 0) / ourCount;
    const chargeCount = ourTeam.filter(d => d.pose === "charge").length;
    const attackCount = ourTeam.filter(d => d.pose === "attack").length;
    if (attackCount > 0) {
      return { msg: `🎯 Attack motion detected (${attackCount}/${ourCount}) — keep momentum.`, mode: "single-team" };
    }
    if (chargeCount === ourCount && ourCount > 0) {
      return { msg: `⚡ All visible players charging — coordinated attack imminent.`, mode: "single-team" };
    }
    if (ourAvgX > 250) {
      return { msg: `🏃 You are advancing into enemy zone — maintain pressure.`, mode: "single-team" };
    }
    return { msg: `📍 ${ourCount} player(s) visible on our side. Enemy out of frame.`, mode: "single-team" };
  }

  if (ourCount === 0) {
    const enemyAvgX = enemyTeam.reduce((a, d) => a + d.x, 0) / enemyCount;
    if (enemyAvgX < 380) {
      return { msg: `⚠️ Enemy advancing (${enemyCount} visible) — reposition our players into frame!`, mode: "single-team" };
    }
    return { msg: `👀 Only enemy visible (${enemyCount}). Adjust camera or move into frame.`, mode: "single-team" };
  }

  // ── Both teams present (1v1 minimum) ──
  // Sample size adapter: 1 vs 1 needs absolute checks; larger teams use ratios
  const small = ourCount < 2 || enemyCount < 2;
  const visibilityNote = (ourCount + enemyCount) < 6
    ? ` [${ourCount}+${enemyCount}/6 visible]` : "";

  // Scenario A: enemy right-side concentration
  const enemyRightCount = enemyTeam.filter(d => d.y > COURT_MID_Y).length;
  const ourRightCount   = ourTeam.filter(d => d.y > COURT_MID_Y).length;
  const enemyRightRatio = enemyRightCount / enemyCount;
  const ourRightRatio   = ourRightCount   / ourCount;

  // For small samples (1-2 per side), require absolute majority instead of strict 0.7
  const rightThreshold = small ? 0.5 : 0.7;
  const ourLowThreshold = small ? 0.5 : 0.4;

  if (enemyRightRatio >= rightThreshold && ourRightRatio <= ourLowThreshold) {
    return {
      msg: `🚨 Enemy on right side (${enemyRightCount}/${enemyCount}) — reinforce right defense!${visibilityNote}`,
      mode: small ? "small-sample" : "full",
    };
  }

  // Scenario B: enemy forward pressure
  const enemyAvgX = enemyTeam.reduce((a, d) => a + d.x, 0) / enemyCount;
  const ourAvgX   = ourTeam.reduce((a, d) => a + d.x, 0)   / ourCount;
  if (enemyAvgX < 380 && ourAvgX < 100) {
    return {
      msg: `⚠️ Heavy forward pressure (enemy avg X=${Math.round(enemyAvgX)}) — deploy shields!${visibilityNote}`,
      mode: small ? "small-sample" : "full",
    };
  }

  // Scenario C (new, small-sample friendly): close-range duel
  if (small) {
    const minDist = Math.min(...ourTeam.map(o =>
      Math.min(...enemyTeam.map(e =>
        Math.hypot(o.x - e.x, o.y - e.y)
      ))
    ));
    if (minDist < 120) {
      return {
        msg: `⚔️ Close-range engagement detected (${Math.round(minDist)}px) — react fast!${visibilityNote}`,
        mode: "small-sample",
      };
    }
  }

  return {
    msg: `✅ Formation stable.${visibilityNote}`,
    mode: small ? "small-sample" : "full",
  };
}

// ────────────────────────────────────────────────────────────────
//  Socket Connections
// ────────────────────────────────────────────────────────────────
io.on('connection', (socket) => {
  console.log(`🔌 Device connected: ${socket.id}`);

  socket.on('hado_stream', (incomingData) => {
    const t = Date.now();
    const stamped = incomingData.map(p => ({ ...p, timestamp: t }));
    dataBuffer.push(...stamped);
    dataBuffer = dataBuffer.filter(d => (t - d.timestamp) < BUFFER_WINDOW);

    // Forward raw frame to dashboards
    socket.broadcast.emit('dashboard_update', incomingData);
  });

  socket.on('disconnect', () => {
    console.log(`❎ Device disconnected: ${socket.id}`);
  });
});

// ────────────────────────────────────────────────────────────────
//  Main analysis loop
// ────────────────────────────────────────────────────────────────
setInterval(() => {
  // ─── Detection status (always emit, even with 0 players) ───
  const uniquePids = new Set(dataBuffer.map(d => d.playerId));
  const teamA = new Set(dataBuffer.filter(d => d.x <  COURT_CENTER_X).map(d => d.playerId)).size;
  const teamB = new Set(dataBuffer.filter(d => d.x >= COURT_CENTER_X).map(d => d.playerId)).size;
  const detected = uniquePids.size;
  const quality  = detected >= 5 ? "good"
                  : detected >= 3 ? "partial"
                  : detected >= 1 ? "low"
                  : "none";

  io.emit('detection_status', {
    detected, expected: 6, teamA, teamB, quality,
  });

  if (dataBuffer.length === 0) {
    io.emit('tactical_advice', { msg: "👁️ Waiting for players...", mode: "idle" });
    return;
  }

  // ─── Hybrid role classification ───
  const { roles, coldStart, elapsed } = classifyRoles(dataBuffer);
  io.emit('player_roles', { roles, coldStart, elapsedMs: elapsed });

  // ─── Tactical scenarios (graceful with 1+ player) ───
  const advice = analyzeTactics(dataBuffer);
  if (advice) {
    io.emit('tactical_advice', advice);
    console.log(`[${new Date().toISOString()}] [${quality}|A${teamA}/B${teamB}] ${advice.msg}`);
  }

  // Console role summary
  if (Object.keys(roles).length > 0) {
    const summary = Object.entries(roles)
      .map(([pid, r]) =>
        `P${pid}(${r.team})=${r.role}[${r.source}|${r.confidence.toFixed(2)}]`)
      .join("  ");
    console.log(`  ↳ ${coldStart ? "🧊 Cold" : "🔥 Warm"} ${summary}`);
  }
}, ANALYSIS_PERIOD);

// ────────────────────────────────────────────────────────────────
//  Start Server
// ────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`🚀 HADO backend running on port ${PORT}`);
  console.log(`   Buffer window:   ${BUFFER_WINDOW / 1000}s`);
  console.log(`   Analysis cycle:  ${ANALYSIS_PERIOD / 1000}s`);
  console.log(`   Cold-start span: ${COLD_START_MS / 1000}s (color-dominant)`);
  console.log(`   Behavior window: ${ROLE_WINDOW / 1000}s`);
});
