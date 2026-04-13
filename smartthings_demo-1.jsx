import { useState } from "react";

const T = {
  bg: "#07090e", card: "#0c1018", border: "#161d2a",
  red: "#f87171", green: "#4ade80", blue: "#60a5fa", cyan: "#22d3ee",
  amber: "#fbbf24", violet: "#a78bfa", pink: "#f472b6", orange: "#fb923c",
  white: "#f1f5f9", dim: "#64748b",
  samsung: "#6366f1",
};

function Txt({ children, size = 13, color = "#b8c0cc", bold = false, align }) {
  return <p style={{ fontFamily: "'Helvetica Neue', sans-serif", fontSize: size, color, lineHeight: 1.6, margin: "0 0 4px", fontWeight: bold ? 700 : 400, textAlign: align }}>{children}</p>;
}
function Code({ children }) {
  return <pre style={{ fontFamily: "monospace", fontSize: 10, color: "#b8c0cc", lineHeight: 1.65, margin: "4px 0 0", whiteSpace: "pre-wrap", padding: "8px 10px", background: "#060810", borderRadius: 6, border: `1px solid ${T.border}` }}>{children}</pre>;
}
function Box({ children, color = T.border, glow = false }) {
  return <div style={{ padding: "12px 14px", background: T.card, borderRadius: 10, border: `1px solid ${color}25`, boxShadow: glow ? `0 0 20px ${color}10` : "none", marginBottom: 8 }}>{children}</div>;
}
function Result({ children, color, label }) {
  return <div style={{ padding: "10px 14px", background: `${color}06`, borderRadius: 10, borderLeft: `3px solid ${color}50`, marginBottom: 8 }}><div style={{ fontFamily: "monospace", fontSize: 9, color, letterSpacing: 2, marginBottom: 4, fontWeight: 600 }}>{label}</div><pre style={{ fontFamily: "monospace", fontSize: 10, color: "#c8ccd4", lineHeight: 1.6, margin: 0, whiteSpace: "pre-wrap" }}>{children}</pre></div>;
}
function Tag({ children, color }) {
  return <div style={{ display: "inline-block", padding: "4px 14px", background: `${color}12`, border: `1px solid ${color}30`, borderRadius: 20, fontFamily: "monospace", fontSize: 9, letterSpacing: 3, color, textTransform: "uppercase", marginBottom: 10 }}>{children}</div>;
}
function Mod({ mod, name, label, color, crown }) {
  return <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}><div style={{ width: 34, height: 34, borderRadius: 8, background: `${color}15`, border: `1.5px solid ${color}40`, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "monospace", fontSize: 10, fontWeight: 700, color }}>{mod}</div><div><div style={{ fontFamily: "'Helvetica Neue', sans-serif", fontSize: 17, fontWeight: 700, color: T.white }}>{name} {crown ? "👑" : ""}</div><div style={{ fontSize: 10, color: T.dim }}>{label}</div></div></div>;
}
function Eng({ color, title, lines }) {
  return <div style={{ padding: "8px 12px", background: T.card, borderRadius: 8, borderLeft: `3px solid ${color}50`, marginBottom: 6 }}><div style={{ fontFamily: "monospace", fontSize: 9, color, fontWeight: 600, letterSpacing: 1, marginBottom: 4 }}>{title}</div>{lines.map((l, i) => <div key={i} style={{ fontFamily: i === lines.length - 1 ? "sans-serif" : "monospace", fontSize: i === lines.length - 1 ? 11 : 10, color: i === lines.length - 1 ? color : "#94a3b8", fontWeight: i === lines.length - 1 ? 600 : 400, marginTop: i === lines.length - 1 ? 4 : 1 }}>{l}</div>)}</div>;
}
function Axis({ num, name, detail, score, color }) {
  return <div style={{ padding: "8px 12px", background: T.card, borderRadius: 8, borderLeft: `3px solid ${color}50`, display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}><div style={{ flex: 1 }}><div style={{ fontFamily: "monospace", fontSize: 9, color, fontWeight: 600 }}>AXIS {num}: {name}</div><div style={{ fontSize: 10, color: T.dim, marginTop: 2 }}>{detail}</div></div><div style={{ fontFamily: "monospace", fontSize: 18, fontWeight: 700, color }}>{score}</div></div>;
}
function Sp({ h = 8 }) { return <div style={{ height: h }} />; }

const slides = [
  // 1 TITLE
  () => (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", padding: "0 24px", textAlign: "center" }}>
      <Tag color={T.samsung}>SMARTTHINGS USE CASE</Tag>
      <Sp h={8} />
      <div style={{ fontSize: 30, fontWeight: 700, color: T.white, lineHeight: 1.3 }}>The Invisible{"\n"}Energy Thief</div>
      <Sp h={12} />
      <Txt size={13} color={T.dim}>A $2 sensor fails. 47 devices report green.</Txt>
      <Txt size={13} color={T.dim}>Electricity bill spikes 40%. Nobody knows why.</Txt>
      <Sp h={24} />
      <Box>
        <Txt size={11} color={T.dim}>One SmartThings scenario through all 8 AMACE modules.</Txt>
        <Txt size={11} color={T.dim}>Every number is real. Every step verifiable.</Txt>
      </Box>
    </div>
  ),

  // 2 THE HOME
  () => (
    <div style={{ padding: "20px 16px 90px" }}>
      <Tag color={T.amber}>THE SETUP</Tag>
      <div style={{ fontSize: 20, fontWeight: 700, color: T.white, marginBottom: 10 }}>47 Devices, One Home</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, marginBottom: 12 }}>
        {[
          { icon: "🏠", name: "SmartThings Hub", d: "Zigbee+Z-Wave+WiFi+Matter", c: T.samsung },
          { icon: "❄️", name: "WindFree AC", d: "AI Energy Mode", c: T.cyan },
          { icon: "🧊", name: "Bespoke Fridge", d: "Ambient temp controlled", c: T.blue },
          { icon: "💡", name: "12 Lights", d: "IKEA+Eve, Matter", c: T.amber },
          { icon: "📷", name: "3 Cameras", d: "Aqara, Matter 1.5", c: T.green },
          { icon: "⌚", name: "Galaxy Watch", d: "Sleep + health", c: T.pink },
        ].map(d => (
          <div key={d.name} style={{ padding: "8px", background: T.card, borderRadius: 8, border: `1px solid ${T.border}`, textAlign: "center" }}>
            <div style={{ fontSize: 20 }}>{d.icon}</div>
            <div style={{ fontSize: 10, fontWeight: 600, color: d.c, marginTop: 2 }}>{d.name}</div>
            <div style={{ fontSize: 8, color: T.dim, marginTop: 1 }}>{d.d}</div>
          </div>
        ))}
      </div>
      <Box color={T.red}>
        <Txt size={11} color={T.red} bold>THE INVISIBLE PROBLEM</Txt>
        <Sp h={4} />
        <Txt>Hub sensor reads <span style={{ fontWeight: 700, color: T.red }}>18°C</span>. Actual: <span style={{ fontWeight: 700, color: T.white }}>21°C</span>.</Txt>
        <Txt>3°C error. Within spec. No fault alert.</Txt>
        <Txt>AC trusts it. Fridge trusts it. Routine engine trusts it.</Txt>
        <Txt bold color={T.red}>One sensor. Three cascading failures. Zero alerts.</Txt>
      </Box>
    </div>
  ),

  // 3 THREE STRATA
  () => (
    <div style={{ padding: "20px 16px 90px" }}>
      <Tag color={T.dim}>DATA STRATA</Tag>
      <div style={{ fontSize: 20, fontWeight: 700, color: T.white, marginBottom: 10 }}>Three Layers of Smart Home Data</div>
      <Box color={T.red}>
        <Txt size={11} color={T.red} bold>S₀ — RAW DEVICE TELEMETRY</Txt>
        <Code>{"Hub temp sensor:   18°C (actual: 21°C)\nAC duty cycle:     ON 42min / OFF 18min\nFridge compressor: ON/OFF timestamps\nSmart plug watts:  per-device readings"}</Code>
      </Box>
      <Box color={T.amber}>
        <Txt size={11} color={T.amber} bold>S₁ — COMPUTED FEATURES</Txt>
        <Code>{"AC energy/hour:          +28% above baseline\nFridge restart freq:     14/hr (baseline 8/hr)\nHousehold consumption:   +40% overall\nSleep environment:       degraded (too cold)"}</Code>
      </Box>
      <Box color={T.violet}>
        <Txt size={11} color={T.violet} bold>S₂ — AI MODEL OUTPUTS</Txt>
        <Code>{"AI Energy Mode: \"extend AC cycles\" (wrong!)\nRoutine engine: confused (conflicted signal)\nInsurance risk: \"elevated stress\" (HSB flag)"}</Code>
      </Box>
      <Result color={T.dim} label="THE BLIND SPOT">{"Each layer says \"normal.\" The problem exists BETWEEN layers.\nHub sensor (S₀) corrupts features (S₁) which corrupts AI (S₂)."}</Result>
    </div>
  ),

  // 4 M1 SSIE
  () => (
    <div style={{ padding: "20px 16px 90px" }}>
      <Mod mod="M1" name="SSIE" label="Schema Inference" color={T.cyan} />
      <Txt>SmartThings streams connect. SSIE examines every field:</Txt>
      <Sp />
      <div style={{ borderRadius: 8, overflow: "hidden", border: `1px solid ${T.border}` }}>
        <div style={{ display: "flex", background: `${T.cyan}10`, padding: "5px 8px" }}>
          {["Field", "Signal", "Verdict"].map(h => <div key={h} style={{ flex: 1, fontFamily: "monospace", fontSize: 9, fontWeight: 600, color: T.cyan }}>{h}</div>)}
        </div>
        {[
          ["device_id", "High entropy, 47 unique", "ENTITY"],
          ["room", "Low cardinality, 6 values", "DIMENSION"],
          ["protocol", "4 types, stable", "DIMENSION"],
          ["timestamp", "Monotonic, ISO8601", "TIMESTAMP"],
          ["watt_hours", "Continuous, high var", "METRIC"],
          ["temp_reading", "Continuous, moderate", "METRIC"],
        ].map((r, i) => (
          <div key={i} style={{ display: "flex", padding: "4px 8px", background: i % 2 === 0 ? T.card : T.bg, borderTop: `1px solid ${T.border}` }}>
            {r.map((c, j) => <div key={j} style={{ flex: 1, fontSize: 10, color: j === 2 ? T.cyan : "#b8c0cc" }}>{c}</div>)}
          </div>
        ))}
      </div>
      <Sp h={10} />
      <Box color={T.cyan}>
        <Txt size={11} color={T.cyan} bold>CRITICAL DISCOVERY</Txt>
        <Txt size={12}>SSIE finds that <span style={{ fontWeight: 700, color: T.cyan }}>hub_sensor_007</span> in raw telemetry = <span style={{ fontWeight: 700, color: T.cyan }}>ambient_source</span> in AC config = <span style={{ fontWeight: 700, color: T.cyan }}>room_temp_device</span> in fridge logic.</Txt>
        <Txt size={12}>Three strata. Three field names. Same physical sensor.</Txt>
      </Box>
      <Result color={T.cyan} label="OUTPUT → M2">{"Schema Map: 47 entities, 6 rooms, 4 protocols\nIdentity link: hub_sensor_007 = AC.ambient = fridge.room_temp\nTime: 2.1 seconds. Config: zero."}</Result>
    </div>
  ),

  // 5 M2 SADD
  () => (
    <div style={{ padding: "20px 16px 90px" }}>
      <Mod mod="M2" name="SADD" label="Deviation Detection" color={T.amber} />
      <Txt>Three engines fire independently:</Txt>
      <Sp />
      <Eng color={T.red} title="S₀ ENGINE: Hub Temperature" lines={[
        "30-day median: 21.2°C | MAD: 0.4°C",
        "Current: 18.0°C",
        "Z = |18.0 − 21.2| / 0.4 = 8.0",
        "Threshold k=4.0 (self-calibrated)",
        "8.0 > 4.0 → DEVIATION DETECTED",
      ]} />
      <Eng color={T.amber} title="S₁ ENGINE: AC Energy" lines={[
        "Baseline: 2.1 kWh/hr ± 0.3",
        "Current: 2.7 kWh/hr ± 0.6",
        "PSI = 0.87 (threshold: 0.20)",
        "0.87 > 0.20 → SHIFT DETECTED",
      ]} />
      <Eng color={T.violet} title="S₂ ENGINE: AI Energy Mode" lines={[
        "Baseline: 'extend cycle' 12% of time",
        "Current: 'extend cycle' 67% of time",
        "67% outside normal interval (threshold 15%)",
        "67% > 15% → DRIFT DETECTED",
      ]} />
      <Sp />
      <Result color={T.amber} label="OUTPUT: 3 DSTs → M3">{"DST#1 (S₀): hub_temp | z=−8.0 | kitchen\nDST#2 (S₁): ac_energy | PSI=0.87 | kitchen\nDST#3 (S₂): energy_mode | 67% breach | kitchen\n\nNo device alert ever fired. All within individual specs."}</Result>
    </div>
  ),

  // 6 M3 ERD
  () => (
    <div style={{ padding: "20px 16px 90px" }}>
      <Mod mod="M3" name="ERD" label="Dimensional Search" color={T.violet} />
      <Txt>Which devices are affected?</Txt>
      <Txt size={11} color={T.dim}>Dimensions: room(6) x type(12) x protocol(4) x mfg(8) x source(5) = 11,520 slices</Txt>
      <Sp />
      <Box color={T.violet}>
        <Txt size={10} color={T.dim}>BEAM SEARCH TRACE</Txt>
        <Sp h={4} />
        {[
          { d: "1", s: "room = kitchen", c: "4.2x", sc: "8,421", r: "✓", col: T.violet },
          { d: "2", s: "kitchen + climate_sensor", c: "7.8x", sc: "12,847", r: "✓✓", col: T.violet },
          { d: "3", s: "kitchen + climate + hub_env", c: "12.1x", sc: "18,234", r: "✓✓✓ BEST", col: T.green },
          { d: "4", s: "+ protocol = zigbee", c: "12.3x", sc: "6,102", r: "✗ STOP", col: T.red },
        ].map((b, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 0", borderBottom: `1px solid ${T.border}` }}>
            <div style={{ fontFamily: "monospace", fontSize: 9, color: T.dim, minWidth: 16 }}>{b.d}</div>
            <div style={{ flex: 1, fontSize: 10, color: "#b8c0cc" }}>{b.s}</div>
            <div style={{ fontFamily: "monospace", fontSize: 9, color: T.dim }}>{b.c}</div>
            <div style={{ fontFamily: "monospace", fontSize: 9, color: T.dim, minWidth: 44, textAlign: "right" }}>{b.sc}</div>
            <div style={{ fontFamily: "monospace", fontSize: 9, color: b.col, minWidth: 64, textAlign: "right", fontWeight: 600 }}>{b.r}</div>
          </div>
        ))}
      </Box>
      <Sp />
      <Result color={T.violet} label="OUTPUT → M4">{"Best slice: kitchen × climate_sensor × hub_environmental\nConcentration: 12.1x above baseline\nMeaning: \"anomaly is in the hub's kitchen temp sensor\"\nChecked: 31 of 11,520 → 99.73% pruned"}</Result>
    </div>
  ),

  // 7 M4 CSCT
  () => (
    <div style={{ padding: "20px 16px 90px" }}>
      <Mod mod="M4" name="CSCT" label="Cross-Stratum Correlation" color={T.cyan} />
      <Txt>Testing: Hub sensor drift (S₀) vs AC energy shift (S₁)</Txt>
      <Sp />
      <Axis num="1" name="Entity Coincidence" detail="Hub directly feeds AC. 100% entity overlap." score="0.98" color={T.blue} />
      <Axis num="2" name="Temporal Precedence" detail="Sensor drifted 4hr before AC shifted. Correct direction." score="0.94" color={T.green} />
      <Axis num="3" name="Dimensional Coherence" detail="Both in kitchen. Exact match." score="1.00" color={T.amber} />
      <Axis num="4" name="Semantic Affinity" detail="ambient_temp → hvac_duty: trained similarity 0.86" score="0.86" color={T.violet} />
      <Sp />
      <Result color={T.cyan} label="COMBINED SCORE">{"0.38×0.98 + 0.28×0.94 + 0.14×1.00 + 0.20×0.86\n= 0.372 + 0.263 + 0.140 + 0.172 = 0.947\n\nThreshold: 0.70 → CORRELATED\n\nSecond chain: hub → fridge (score: 0.921)\nBoth chains share same root: hub_sensor_007"}</Result>
    </div>
  ),

  // 8 M5 CAP
  () => (
    <div style={{ padding: "20px 16px 90px" }}>
      <Mod mod="M5" name="CAP" label="Causal Proof" color={T.green} crown />
      <Txt>Is the hub sensor the genuine CAUSE?</Txt>
      <Sp />
      <Box color={T.green} glow>
        <Txt size={11} color={T.green} bold>TEST 1: EXCLUSIVITY</Txt>
        <Txt size={10} color={T.dim}>"Remove hub sensor data. Does AC anomaly disappear?"</Txt>
        <Code>{"Replace hub readings with real temp (21°C).\nAC anomaly: 0.87 PSI → 0.04 PSI (normal)\nReduction: 95.4%  |  95% CI: [92.1%, 97.8%]\nThreshold: >30% → PASS ✓"}</Code>
      </Box>
      <Box color={T.green}>
        <Txt size={11} color={T.green} bold>TEST 2: SUFFICIENCY</Txt>
        <Txt size={10} color={T.dim}>"Does hub error alone predict AC overconsumption?"</Txt>
        <Code>{"3°C error → model predicts +26% longer cycles\nActual: +28%  |  Error: 7.1%\nThreshold: <50% → PASS ✓"}</Code>
      </Box>
      <Box color={T.green}>
        <Txt size={11} color={T.green} bold>TEST 3: MINIMALITY</Txt>
        <Txt size={10} color={T.dim}>"Are both chains needed?"</Txt>
        <Code>{"AC chain alone: explains 68% of energy spike\nFridge chain alone: explains 12%\nBoth together: explains 96%\nBoth NECESSARY → PASS ✓"}</Code>
      </Box>
      <Result color={T.green} label="VERDICT">{"CAUSAL at p < 0.05. Confidence: 0.94\nHub sensor drift causes BOTH chains.\nThis is proof, not correlation."}</Result>
    </div>
  ),

  // 9 M6+M7+M8
  () => (
    <div style={{ padding: "20px 16px 90px" }}>
      <Tag color={T.pink}>MODULES 6 + 7 + 8</Tag>
      <div style={{ fontSize: 18, fontWeight: 700, color: T.white, marginBottom: 10 }}>Story + Prediction + Explanation</div>
      <Box color={T.pink}>
        <Txt size={11} color={T.pink} bold>M6 CIGS — Complete Story</Txt>
        <Code>{"ROOT CAUSE: Hub sensor (hub_sensor_007)\n  reads 18°C (actual 21°C)\n\nCHAIN 1: drift → AC +28% cycles → AI extends more\nCHAIN 2: drift → fridge reduces duty → restarts +75%\n\nIMPACT: +40% energy ($67/month)\nAFFECTED: all climate devices in kitchen\nCONFIDENCE: 0.94"}</Code>
      </Box>
      <Box color={T.orange}>
        <Txt size={11} color={T.orange} bold>M7 PPE — Anti-Correlation Detected</Txt>
        <Code>{"MISSING SIGNAL:\nRoutine engine SHOULD have suggested\n\"turn off AC at night\" (did 4x before).\n\nIt DID NOT. Why?\n→ Routine engine ALSO trusts the hub sensor.\n→ Sees 18°C → thinks cooling IS appropriate.\n\nFINDING: 3 systems contaminated by 1 sensor."}</Code>
      </Box>
      <Box color={T.violet}>
        <Txt size={11} color={T.violet} bold>M8 CIL — Why + Past Fix</Txt>
        <Code>{"RETRIEVED: Hub spec Sec 4.3\n\"NTC thermistor calibration: monthly\"\nLast calibration: 26 days ago.\n\nRESOLUTION (from HVAC-INC-2025-0034):\n\"Replace sensor + add redundant temp check\"\nCost: $15 + 30 min. Recovery: 22 min."}</Code>
      </Box>
    </div>
  ),

  // 10 FINAL OUTPUT
  () => (
    <div style={{ padding: "20px 16px 90px" }}>
      <Tag color={T.green}>FINAL OUTPUT</Tag>
      <div style={{ fontSize: 18, fontWeight: 700, color: T.white, marginBottom: 10 }}>What the Homeowner Sees</div>
      <div style={{ padding: "14px 16px", background: "#0a120a", borderRadius: 12, border: `1px solid ${T.green}25`, boxShadow: `0 0 24px ${T.green}08` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
          <div style={{ width: 8, height: 8, borderRadius: 4, background: T.green, boxShadow: `0 0 8px ${T.green}` }} />
          <span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 700, color: T.green, letterSpacing: 2 }}>SMARTTHINGS INSIGHT</span>
          <span style={{ fontFamily: "monospace", fontSize: 10, color: T.dim, marginLeft: "auto" }}>94%</span>
        </div>
        <Code>{`ROOT CAUSE:
Hub temp sensor reads 3°C below actual.

THIS CAUSED:
1. WindFree AC running 28% longer
2. Fridge compressor restarting 75% more
3. Routine engine blind to the problem

IMPACT: +40% energy ($67/month)

FIX: Replace hub sensor ($15, 30 min)
     Add secondary sensor for cross-check

PROOF: Removing sensor influence eliminates
95.4% of energy anomaly (95% confidence)

PRECEDENT: HVAC-INC-2025-0034 (same fix)`}</Code>
      </div>
      <Sp h={12} />
      <div style={{ display: "flex", gap: 8 }}>
        {[
          { v: "4 min", l: "Detection", s: "vs never", c: T.green },
          { v: "$804/yr", l: "Savings", s: "recovered", c: T.amber },
          { v: "Zero", l: "Config", s: "plug & play", c: T.cyan },
        ].map(s => (
          <div key={s.l} style={{ flex: 1, padding: "10px 8px", background: T.card, borderRadius: 10, border: `1px solid ${T.border}`, textAlign: "center" }}>
            <div style={{ fontSize: 18, fontWeight: 700, color: s.c }}>{s.v}</div>
            <div style={{ fontSize: 10, fontWeight: 600, color: T.white, marginTop: 2 }}>{s.l}</div>
            <div style={{ fontSize: 8, color: T.dim }}>{s.s}</div>
          </div>
        ))}
      </div>
    </div>
  ),

  // 11 WHY NOVEL
  () => (
    <div style={{ padding: "20px 16px 90px" }}>
      <Tag color={T.samsung}>WHY THIS IS PATENTABLE</Tag>
      <div style={{ fontSize: 18, fontWeight: 700, color: T.white, marginBottom: 10 }}>What No Existing Tool Can Do</div>
      {[
        { what: "SmartThings Energy", cant: "Shows total is up. Can't identify which device INTERACTION is the cause.", c: T.red },
        { what: "Device diagnostics", cant: "Each device: \"normal.\" AC isn't broken. Fridge isn't broken. Sensor within spec.", c: T.red },
        { what: "Manual investigation", cant: "Check each device separately. Weeks to connect hub → AC → fridge → routine engine.", c: T.red },
      ].map((x, i) => (
        <Box key={i} color={x.c}>
          <Txt size={11} color={x.c} bold>{"❌ " + x.what}</Txt>
          <Txt size={11}>{x.cant}</Txt>
        </Box>
      ))}
      <Sp h={4} />
      <Box color={T.green} glow>
        <Txt size={11} color={T.green} bold>{"✅ AMACE"}</Txt>
        <Txt size={11}>Correlates across Matter/Zigbee/WiFi boundaries. Traces one sensor through two parallel causal chains across three strata. Proves causation via ablation. Finds the anti-correlation. Explains via hardware spec. 4 minutes. Zero config.</Txt>
      </Box>
      <Sp h={8} />
      <Result color={T.samsung} label="PATENT CLAIM">{"\"...wherein the observation stratum comprises\ndevice telemetry from a smart home platform\nsupporting Matter, Zigbee, Z-Wave, and WiFi,\nthe representation stratum comprises energy\nprofiles and device health metrics, and the\ninference stratum comprises AI-driven automation\noptimization decisions.\""}</Result>
    </div>
  ),

  // 12 CLOSE
  () => (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", padding: "0 24px", textAlign: "center" }}>
      <Tag color={T.green}>SMARTTHINGS + AMACE</Tag>
      <Sp h={16} />
      <div style={{ fontSize: 24, fontWeight: 700, color: T.white }}>A $2 sensor failed.</div>
      <Sp h={8} />
      <Txt color={T.dim}>47 devices said "all clear."</Txt>
      <Txt color={T.dim}>Energy bill spiked 40%.</Txt>
      <Txt color={T.dim}>Three systems contaminated.</Txt>
      <Txt color={T.dim}>No existing tool could see it.</Txt>
      <Sp h={20} />
      <Box color={T.green} glow>
        <Txt size={14} color={T.green} bold align="center">AMACE found the root cause in 4 minutes,</Txt>
        <Txt size={14} color={T.green} bold align="center">proved it at 95% confidence,</Txt>
        <Txt size={14} color={T.green} bold align="center">explained WHY, and cited the fix.</Txt>
      </Box>
      <Sp h={16} />
      <Txt size={11} color={T.dim}>430 million SmartThings users.</Txt>
      <Txt size={11} color={T.dim}>Every home is a potential AMACE deployment.</Txt>
    </div>
  ),
];

export default function App() {
  const [cur, setCur] = useState(0);
  const Slide = slides[cur];
  return (
    <div style={{ height: "100vh", background: T.bg, position: "relative", overflow: "hidden" }}>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "#ffffff06", zIndex: 50 }}>
        <div style={{ height: "100%", background: `linear-gradient(90deg, ${T.samsung}, ${T.green})`, width: `${((cur + 1) / slides.length) * 100}%`, transition: "width 0.4s" }} />
      </div>
      <div style={{ position: "absolute", top: 8, right: 14, fontFamily: "monospace", fontSize: 10, color: "#ffffff15", zIndex: 50 }}>
        {String(cur + 1).padStart(2, "0")}/{String(slides.length).padStart(2, "0")}
      </div>
      <div style={{ height: "calc(100% - 52px)", overflow: "auto", position: "relative", zIndex: 10 }}>
        <Slide />
      </div>
      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 52, display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 16px", background: `linear-gradient(transparent, ${T.bg})`, zIndex: 50 }}>
        <button onClick={() => cur > 0 && setCur(c => c - 1)} disabled={cur === 0} style={{ padding: "8px 18px", background: "#ffffff08", border: "1px solid #ffffff10", borderRadius: 8, color: T.dim, fontSize: 13, cursor: "pointer", opacity: cur === 0 ? 0.2 : 1 }}>{"←"}</button>
        <div style={{ display: "flex", gap: 4 }}>
          {slides.map((_, i) => (
            <div key={i} onClick={() => setCur(i)} style={{ width: i === cur ? 16 : 5, height: 5, borderRadius: 3, background: i === cur ? T.green : "#ffffff12", cursor: "pointer", transition: "all 0.3s" }} />
          ))}
        </div>
        <button onClick={() => cur < slides.length - 1 && setCur(c => c + 1)} disabled={cur === slides.length - 1} style={{ padding: "8px 18px", background: T.green, border: "none", borderRadius: 8, color: "#000", fontSize: 13, fontWeight: 700, cursor: "pointer", opacity: cur === slides.length - 1 ? 0.3 : 1 }}>{"→"}</button>
      </div>
    </div>
  );
}
