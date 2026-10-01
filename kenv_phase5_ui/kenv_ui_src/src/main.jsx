import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Alert, Button, Card, Collapse, ConfigProvider, Descriptions, Empty, Input, Layout, List, Progress, Select, Space,
  Spin, Switch, Table, Tabs, Tag, Timeline, Tooltip, Typography, theme,
} from "antd";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip as RTip, XAxis, YAxis,
} from "recharts";

const { Text, Title } = Typography;

// ---------------------------------------------------------------- helpers
async function api(path, params = {}) {
  const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ""));
  const r = await fetch(`/api/${path}${q.toString() ? "?" + q : ""}`, { credentials: "same-origin" });
  let body = null;
  try { body = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) throw new Error((body && body.error) || (r.status === 401 ? "Access token missing: open the link that `kenv ui` printed." : `HTTP ${r.status}`));
  return body;
}

function useApi(path, params, enabled = true, pollMs = 0) {
  const [state, set] = useState({ data: null, error: null, loading: enabled });
  const key = path + JSON.stringify(params);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!enabled) return undefined;
    let dead = false;
    set((s) => ({ ...s, loading: s.data === null || tick === 0 }));
    const run = () => api(path, params)
      .then((data) => !dead && set({ data, error: null, loading: false }))
      .catch((error) => !dead && set({ data: null, error: error.message, loading: false }));
    run();
    const id = pollMs ? setInterval(run, pollMs) : null;
    return () => { dead = true; if (id) clearInterval(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled, tick, pollMs]);
  return { ...state, reload: () => setTick((t) => t + 1) };
}

const human = (n) => {
  if (n === null || n === undefined) return "-";
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; let x = Number(n);
  while (x >= 1024 && i < u.length - 1) { x /= 1024; i++; }
  return `${i ? x.toFixed(1) : x.toFixed(0)} ${u[i]}`;
};
const dur = (s) => {
  if (s === null || s === undefined) return "-";
  s = Math.round(s); const h = Math.floor(s / 3600); const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : m ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
};
const when = (iso) => (iso ? new Date(iso).toLocaleString() : "-");
const ago = (iso) => {
  if (!iso) return "never";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)}m ago` : s < 86400 ? `${Math.floor(s / 3600)}h ago` : `${Math.floor(s / 86400)}d ago`;
};
const num = (x) => (typeof x === "number" ? (Number.isInteger(x) ? x : Number(x.toPrecision(5))) : x);
const vlabel = (r) => (r.name ? `${r.version} (${r.name})` : r.version);

function Hash({ h }) {
  if (!h) return <Text type="secondary">-</Text>;
  return <Tooltip title={h}><Text className="mono" copyable={{ text: h }}>{h.slice(0, 12)}…</Text></Tooltip>;
}

function Fail({ error }) { return error ? <Alert type="error" showIcon message={error} /> : null; }
function Wait({ q, children }) {
  if (q.error) return <Fail error={q.error} />;
  if (q.loading && !q.data) return <div style={{ padding: 48, textAlign: "center" }}><Spin /></div>;
  return q.data ? children(q.data) : null;
}

function Chart({ title, unit, kind = "bar", data, x, keys, height = 190 }) {
  const { token } = theme.useToken();
  const colors = [token.colorPrimary, "#d46b08", "#531dab", "#389e0d"];
  const C = kind === "line" ? LineChart : BarChart;
  return (
    <Card size="small" title={<span>{title}{unit ? <Text type="secondary"> {unit}</Text> : null}</span>}>
      {data.length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="no data" /> : (
        <ResponsiveContainer width="100%" height={height}>
          <C data={data} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={token.colorBorderSecondary} />
            <XAxis dataKey={x} tick={{ fontSize: 11, fill: token.colorTextSecondary }} />
            <YAxis tick={{ fontSize: 11, fill: token.colorTextSecondary }} width={52} />
            <RTip contentStyle={{ background: token.colorBgElevated, border: `1px solid ${token.colorBorder}`, fontSize: 12 }} />
            {keys.length > 1 ? <Legend wrapperStyle={{ fontSize: 11 }} /> : null}
            {keys.map((k, i) => (kind === "line"
              ? <Line key={k} type="monotone" dataKey={k} stroke={colors[i % 4]} dot isAnimationActive={false} connectNulls />
              : <Bar key={k} dataKey={k} fill={colors[i % 4]} isAnimationActive={false} />))}
          </C>
        </ResponsiveContainer>)}
    </Card>
  );
}

// ---------------------------------------------------------------- panels
function Versions({ ov, pick }) {
  const cols = [
    { title: "Version", dataIndex: "version", render: (_, r) => <a onClick={() => pick(r.version)}>{vlabel(r)}</a> },
    { title: "Tags", dataIndex: "tags", render: (t, r) => <>{r.active ? <Tag color="green">active</Tag> : null}{t.map((x) => <Tag key={x} color="blue">{x}</Tag>)}{r.auto ? <Tag>auto-save</Tag> : null}</> },
    { title: "Branch", dataIndex: "branch" },
    { title: "Message", dataIndex: "message", ellipsis: true, render: (m, r) => m || (r.committed ? "" : <Text type="secondary">never committed</Text>) },
    { title: "Sessions", dataIndex: "sessions", align: "right" },
    { title: "Time used", dataIndex: "total_s", align: "right", render: dur },
    { title: "Last used", dataIndex: "last_used", render: (t) => <Tooltip title={when(t)}>{ago(t)}</Tooltip> },
    { title: "Metrics", dataIndex: "metrics", render: (m) => Object.entries(m || {}).map(([k, v]) => <Tag key={k}>{k} {num(v)}</Tag>) },
  ];
  return (
    <Space direction="vertical" style={{ width: "100%" }} size="middle">
      <Table size="small" rowKey="version" columns={cols} dataSource={[...ov.versions].reverse()} pagination={false} scroll={{ x: true }} />
      <div className="grid">
        <Card size="small" title="Branches">
          <List size="small" dataSource={ov.branches} renderItem={(b) => (
            <List.Item>
              <Space wrap>
                <Text strong>{b.name}</Text>{b.name === ov.branch ? <Tag color="green">current</Tag> : null}
                {b.parent ? <Text type="secondary">from {b.parent}</Text> : null}
                {b.versions.length ? b.versions.map((v) => <Tag key={v}><a onClick={() => pick(v)}>{v}</a></Tag>) : <Text type="secondary">no commits</Text>}
              </Space>
            </List.Item>)} />
        </Card>
        <Card size="small" title="Tags">
          {ov.tags.length ? <Space wrap>{ov.tags.map((t) => <Tag key={t.name} color="blue"><a onClick={() => pick(t.version)}>{t.name} → {t.version}</a></Tag>)}</Space> : <Text type="secondary">no tags</Text>}
        </Card>
      </div>
    </Space>
  );
}

function Sessions({ d }) {
  const bars = d.sessions.map((s, i) => ({ name: `#${i + 1}`, minutes: Number(((s.duration_s || 0) / 60).toFixed(1)) }));
  const color = { running: "green", ended: "blue", lost: "red" };
  return (
    <div className="grid" style={{ gridTemplateColumns: "minmax(320px,1fr) minmax(320px,1fr)" }}>
      <Card size="small" title={`Session usage · ${dur(d.sessions.reduce((a, s) => a + (s.duration_s || 0), 0))} total`}>
        {d.sessions.length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="no sessions in this version yet" /> : (
          <Timeline items={d.sessions.map((s) => ({
            color: color[s.state],
            children: (
              <Space direction="vertical" size={0}>
                <Text>{when(s.start)} → {s.state === "running" ? "running now" : when(s.end)}</Text>
                <Space size={4} wrap>
                  <Tag color={color[s.state]}>{dur(s.duration_s)}</Tag>
                  <Tag>{s.gpu && s.gpu !== "none" ? `GPU ${s.gpu}` : "CPU"}</Tag>
                  {s.ended_by ? <Tag>ended: {s.ended_by}</Tag> : null}
                  {s.estimated ? <Tag color="orange">end estimated</Tag> : null}
                  {(s.gpu_log || []).length ? <Tag>{s.gpu_log.length} accelerator switch(es)</Tag> : null}
                </Space>
              </Space>),
          }))} />)}
      </Card>
      <Chart title="Minutes per session" data={bars} x="name" keys={["minutes"]} height={260} />
    </div>
  );
}

function Runs({ d, rows }) {
  const metricNames = useMemo(() => [...new Set(Object.keys(d.metrics || {}))], [d]);
  const across = useMemo(() => {
    const names = [...new Set(rows.flatMap((r) => Object.entries(r.metrics || {}).filter(([, v]) => typeof v === "number").map(([k]) => k)))];
    return names.map((n) => ({ n, data: rows.filter((r) => typeof (r.metrics || {})[n] === "number").map((r) => ({ version: r.version, [n]: r.metrics[n] })) }));
  }, [rows]);
  const within = useMemo(() => {
    const by = {};
    [...d.runs].filter((r) => r.kind === "metric" && typeof r.value === "number").reverse().forEach((r) => { (by[r.name] = by[r.name] || []).push({ n: by[r.name].length + 1, value: r.value }); });
    return Object.entries(by);
  }, [d]);
  const cols = [
    { title: "Run", dataIndex: "id", width: 80 },
    { title: "Kind", dataIndex: "kind", render: (k) => <Tag>{k}</Tag> },
    { title: "Label", dataIndex: "label", render: (l, r) => (r.kind === "metric" ? `${r.name} = ${num(r.value)}` : l) },
    { title: "Status", dataIndex: "status", render: (s, r) => <Tooltip title={r.error}><Tag color={s === "failed" ? "red" : "green"}>{s}</Tag></Tooltip> },
    { title: "Started", dataIndex: "started", render: when },
    { title: "Duration", dataIndex: "duration_s", align: "right", render: (s) => (s === undefined ? "" : dur(s)) },
    { title: "RAM peak", dataIndex: "ram_peak_gb", align: "right", render: (x) => (x === undefined ? "" : `${x} GB`) },
    { title: "VRAM peak", dataIndex: "vram_peak_gb", align: "right", render: (x) => (x === undefined ? "" : `${x} GB`) },
  ];
  return (
    <Space direction="vertical" style={{ width: "100%" }} size="middle">
      <Card size="small" title="Metrics of this version">{metricNames.length ? <Space wrap>{metricNames.map((k) => <Tag key={k} color="geekblue">{k}: {num(d.metrics[k])}</Tag>)}</Space> : <Text type="secondary">none recorded (kenv.metric("auc", 0.93))</Text>}</Card>
      {across.length ? <div className="grid">{across.map((c) => <Chart key={c.n} kind="line" title={`${c.n} across versions`} data={c.data} x="version" keys={[c.n]} />)}</div> : null}
      {within.length ? <div className="grid">{within.map(([n, data]) => <Chart key={n} kind="line" title={`${n} within ${d.version}`} unit="(by run)" data={data} x="n" keys={["value"]} />)}</div> : null}
      <Table size="small" rowKey="id" columns={cols} dataSource={d.runs} pagination={{ pageSize: 15, hideOnSinglePage: true }} scroll={{ x: true }} locale={{ emptyText: "no runs recorded yet" }} />
    </Space>
  );
}

function Resources({ d, rows }) {
  const per = (key) => rows.filter((r) => r.resources && r.resources[key] !== undefined).map((r) => ({ version: r.version, [key]: r.resources[key] }));
  const timed = d.runs.filter((r) => r.kind === "timed").reverse().map((r) => ({ run: r.id, label: r.label, ...r }));
  const res = d.resources || {};
  return (
    <Space direction="vertical" style={{ width: "100%" }} size="middle">
      <Descriptions size="small" bordered column={{ xs: 1, sm: 2, lg: 5 }} title={`Last session peaks · ${d.version}`}>
        <Descriptions.Item label="RAM">{res.ram_peak_gb ?? "-"} GB</Descriptions.Item>
        <Descriptions.Item label="VRAM">{res.vram_peak_gb ?? "-"} GB</Descriptions.Item>
        <Descriptions.Item label="CPU">{res.cpu_peak_pct ?? "-"} %</Descriptions.Item>
        <Descriptions.Item label="Disk">{res.disk_peak_gb ?? "-"} GB</Descriptions.Item>
        <Descriptions.Item label="Runtime">{res.runtime_min ?? "-"} min</Descriptions.Item>
      </Descriptions>
      {rows.length > 1 ? (<>
        <Title level={5} style={{ margin: 0 }}>Across versions</Title>
        <div className="grid">
          <Chart title="RAM peak" unit="GB" data={per("ram_peak_gb")} x="version" keys={["ram_peak_gb"]} />
          <Chart title="VRAM peak" unit="GB" data={per("vram_peak_gb")} x="version" keys={["vram_peak_gb"]} />
          <Chart title="CPU peak" unit="%" data={per("cpu_peak_pct")} x="version" keys={["cpu_peak_pct"]} />
          <Chart title="Disk peak" unit="GB" data={per("disk_peak_gb")} x="version" keys={["disk_peak_gb"]} />
          <Chart title="Runtime" unit="min" data={per("runtime_min")} x="version" keys={["runtime_min"]} />
        </div></>) : null}
      <Title level={5} style={{ margin: 0 }}>Timed runs in {d.version} (kenv.time_start / time_end)</Title>
      {timed.length === 0 ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="no timed runs in this version" /> : (
        <div className="grid">
          <Chart title="RAM peak" unit="GB" data={timed} x="run" keys={["ram_peak_gb"]} />
          <Chart title="VRAM peak" unit="GB" data={timed} x="run" keys={["vram_peak_gb"]} />
          <Chart title="CPU average" unit="%" data={timed} x="run" keys={["cpu_avg_pct"]} />
          <Chart title="Disk change" unit="MB" data={timed} x="run" keys={["disk_delta_mb"]} />
          <Chart title="Duration" unit="s" data={timed} x="run" keys={["duration_s"]} />
        </div>)}
    </Space>
  );
}

function IO({ d }) {
  const out = [
    { title: "Output file", dataIndex: "path", render: (p) => <Text className="mono">{p}</Text> },
    { title: "Size", dataIndex: "size", align: "right", render: human, sorter: (a, b) => (a.size || 0) - (b.size || 0) },
    { title: "SHA-256", dataIndex: "sha256", render: (h) => <Hash h={h} /> },
  ];
  const total = d.io.outputs.reduce((a, o) => a + (o.size || 0), 0);
  return (
    <Space direction="vertical" style={{ width: "100%" }} size="middle">
      <div className="grid">
        <Card size="small" title="Inputs · Kaggle datasets">{d.io.datasets.length ? <Space wrap>{d.io.datasets.map((x) => <Tag key={x}>{x}</Tag>)}</Space> : <Text type="secondary">none attached</Text>}</Card>
        <Card size="small" title="Mounted on the kernel (/kaggle/input)">{d.io.mounted.length ? <Space wrap>{d.io.mounted.map((x) => <Tag key={x}>{x}</Tag>)}</Space> : <Text type="secondary">nothing recorded</Text>}</Card>
        <Card size="small" title="Secret references (names only)">{d.secrets.length ? <Space wrap>{d.secrets.map((x) => <Tag key={x} color="gold">{x}</Tag>)}</Space> : <Text type="secondary">none</Text>}<div><Text type="secondary" style={{ fontSize: 12 }}>kenv never stores secret values.</Text></div></Card>
      </div>
      <Table size="small" rowKey="path" columns={out} dataSource={d.io.outputs} pagination={{ pageSize: 20, hideOnSinglePage: true }} scroll={{ x: true }}
        title={() => <Text strong>Outputs pulled back · {d.io.outputs.length} file(s), {human(total)}</Text>} locale={{ emptyText: "no outputs recorded yet" }} />
      <Table size="small" rowKey="path" pagination={{ pageSize: 10, hideOnSinglePage: true }} dataSource={d.code} scroll={{ x: true }}
        title={() => <Text strong>Code snapshot · {d.code.length} file(s)</Text>} locale={{ emptyText: "no code snapshot (version never committed)" }}
        columns={[{ title: "File", dataIndex: "path", render: (p) => <Text className="mono">{p}</Text> },
          { title: "Size", dataIndex: "size", align: "right", render: (s, r) => (r.skipped ? <Tag>too big to copy</Tag> : human(s)) },
          { title: "SHA-256", dataIndex: "sha256", render: (h) => <Hash h={h} /> }]} />
    </Space>
  );
}

function Deps({ d }) {
  const [q, setQ] = useState("");
  const all = Object.entries(d.dependencies.lock).map(([name, version]) => ({ name, version }));
  const rows = all.filter((r) => r.name.toLowerCase().includes(q.toLowerCase()));
  const declared = Object.entries(d.dependencies.declared || {});
  return (
    <Space direction="vertical" style={{ width: "100%" }} size="middle">
      <Space wrap>
        <Input.Search allowClear placeholder="Search packages" style={{ width: 260 }} value={q} onChange={(e) => setQ(e.target.value)} />
        <Text type="secondary">{rows.length} of {all.length} pinned package(s){d.manifest.python ? ` · Python ${d.manifest.python}` : ""}{d.manifest.docker_image ? ` · image ${d.manifest.docker_image}` : ""}</Text>
      </Space>
      {declared.length ? <Alert type="info" showIcon message="Declared by hand" description={<Space wrap>{declared.map(([k, v]) => <Tag key={k}>{k} {String(v)}</Tag>)}</Space>} /> : null}
      <Table size="small" rowKey="name" dataSource={rows} pagination={{ pageSize: 25, hideOnSinglePage: true }}
        locale={{ emptyText: "the dependency lock is empty: no session has recorded packages for this version yet" }}
        columns={[{ title: "Package", dataIndex: "name", sorter: (a, b) => a.name.localeCompare(b.name) }, { title: "Version", dataIndex: "version", render: (v) => <Text className="mono">{v}</Text> }]} />
    </Space>
  );
}

function Mark({ text, q }) {
  if (!q) return text;
  const parts = text.split(new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "i"));
  return parts.map((p, i) => (p.toLowerCase() === q.toLowerCase() ? <mark key={i}>{p}</mark> : p));
}

function Logs({ d, live }) {
  const [q, setQ] = useState(""); const [term, setTerm] = useState(""); const [errors, setErrors] = useState(false);
  const [follow, setFollow] = useState(false); const [limit, setLimit] = useState(300);
  useEffect(() => { const t = setTimeout(() => setTerm(q.trim()), 300); return () => clearTimeout(t); }, [q]);
  const res = useApi("logs", { v: d.version, q: term, errors: errors ? "1" : "0", limit }, true, follow ? 3000 : 0);
  const box = useRef(null);
  useEffect(() => { if (follow && box.current) box.current.scrollTop = box.current.scrollHeight; }, [res.data, follow]);
  const lines = res.data ? res.data.lines : [];
  return (
    <Space direction="vertical" style={{ width: "100%" }} size="small">
      <Space wrap>
        <Input.Search allowClear placeholder="Search the log" style={{ width: 280 }} value={q} onChange={(e) => setQ(e.target.value)} />
        <Switch checked={errors} onChange={setErrors} /><Text>errors only</Text>
        <Switch checked={follow} onChange={setFollow} /><Text>follow{live ? " (a session is running)" : ""}</Text>
        {res.data && res.data.exists ? <Text type="secondary">{res.data.total} matching line(s){res.data.truncated ? " · newest 16 MB searched" : ""}</Text> : null}
      </Space>
      <Fail error={res.error} />
      {res.data && !res.data.exists ? <Empty description="No log for this version yet: logs are recorded while a session for it is online." /> : (
        <div className="logbox mono" ref={box}>
          {res.data && res.data.older > 0 ? <div style={{ textAlign: "center", padding: 4 }}><Button size="small" onClick={() => setLimit(limit + 500)}>Load {Math.min(500, res.data.older)} older lines ({res.data.older} not shown)</Button></div> : null}
          {lines.map((l, i) => (
            <div key={i} className={`logrow${l.src === "error" || l.src === "stderr" ? " err" : ""}`}>
              <span className="logts">{l.ts ? l.ts.replace("T", " ").replace("Z", "") : ""}</span><span className="logsrc">{l.src}</span><span><Mark text={l.text} q={term} /></span>
            </div>))}
          {res.data && lines.length === 0 ? <div style={{ padding: 12 }}><Text type="secondary">no lines</Text></div> : null}
        </div>)}
    </Space>
  );
}

function Patch({ lines }) {
  return <pre className="patch mono">{lines.map((l, i) => <span key={i} className={l.startsWith("+++") || l.startsWith("---") || l.startsWith("@@") ? "hunk" : l[0] === "+" ? "add" : l[0] === "-" ? "del" : ""}>{l || " "}</span>)}</pre>;
}

function Diff({ rows }) {
  const names = rows.map((r) => ({ value: r.version, label: vlabel(r) }));
  const pool = rows.filter((r) => r.committed).length > 1 ? rows.filter((r) => r.committed) : rows;   // default: the two newest snapshots
  const [to, setTo] = useState(pool.length ? pool[pool.length - 1].version : null);
  const [from, setFrom] = useState(pool.length > 1 ? pool[pool.length - 2].version : null);
  const res = useApi("diff", { from, to }, !!(from && to && from !== to));
  return (
    <Space direction="vertical" style={{ width: "100%" }} size="middle">
      <Space wrap><Select style={{ width: 200 }} options={names} value={from} onChange={setFrom} /><Text>→</Text><Select style={{ width: 200 }} options={names} value={to} onChange={setTo} /></Space>
      {from === to ? <Alert type="info" showIcon message="Pick two different versions." /> : null}
      {from !== to ? <Wait q={res}>{(x) => (
        <Space direction="vertical" style={{ width: "100%" }} size="middle">
          {x.snapshots.some((s) => !s) ? <Alert type="warning" showIcon message="A version without a snapshot (never committed) is shown as an empty project; its recorded metrics still count." /> : null}
          <Card size="small" title={`Code: ${x.files.filter((f) => f.status === "added").length} added, ${x.files.filter((f) => f.status === "modified").length} changed, ${x.files.filter((f) => f.status === "removed").length} removed`}>
            {x.files.length === 0 ? <Text type="secondary">identical</Text> : (
              <Collapse size="small" items={x.files.map((f) => ({
                key: f.path,
                label: <Space><Tag color={f.status === "added" ? "green" : f.status === "removed" ? "red" : "blue"}>{f.status}</Tag><Text className="mono">{f.path}</Text>{f.plus !== undefined ? <Text type="secondary">+{f.plus} −{f.minus}</Text> : null}</Space>,
                children: f.binary ? <Text type="secondary">binary or too large to show</Text> : f.patch === null ? <Text type="secondary">only the first 40 files show a patch</Text> : <><Patch lines={f.patch} />{f.cut ? <Text type="secondary">patch cut at 400 lines</Text> : null}</>,
              }))} />)}
          </Card>
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fill,minmax(360px,1fr))" }}>
            <Card size="small" title={`Libraries: ${x.libs.length} difference(s)`}>
              <Table size="small" rowKey="name" pagination={{ pageSize: 8, hideOnSinglePage: true }} dataSource={x.libs} locale={{ emptyText: "identical" }}
                columns={[{ title: "Package", dataIndex: "name" }, { title: "Change", render: (_, r) => (r.from === null ? <Tag color="green">+ {r.to}</Tag> : r.to === null ? <Tag color="red">− {r.from}</Tag> : <Text className="mono">{r.from} → {r.to}</Text>) }]} />
            </Card>
            <Card size="small" title="Metrics">
              <Table size="small" rowKey="name" pagination={false} dataSource={x.metrics} locale={{ emptyText: "none recorded" }}
                columns={[{ title: "Metric", dataIndex: "name" }, { title: x.from_label, dataIndex: "from", render: (v) => (v === null || v === undefined ? "-" : num(v)) },
                  { title: x.to_label, dataIndex: "to", render: (v) => (v === null || v === undefined ? "-" : num(v)) },
                  { title: "Δ", dataIndex: "delta", render: (v) => (v === null ? "" : <Text type={v < 0 ? "success" : v > 0 ? "danger" : undefined}>{v > 0 ? "+" : ""}{num(v)}</Text>) }]} />
              <Text type="secondary" style={{ fontSize: 12 }}>Whether lower or higher is better depends on the metric.</Text>
            </Card>
            <Card size="small" title="Inputs (datasets)">{x.inputs.added.length + x.inputs.removed.length === 0 ? <Text type="secondary">identical</Text> : <Space wrap>{x.inputs.added.map((i) => <Tag color="green" key={i}>+ {i}</Tag>)}{x.inputs.removed.map((i) => <Tag color="red" key={i}>− {i}</Tag>)}</Space>}</Card>
          </div>
        </Space>)}</Wait> : null}
    </Space>
  );
}

function Quota() {
  const res = useApi("quota", {});
  return (
    <Wait q={res}>{(q) => (
      <Space direction="vertical" style={{ width: "100%" }} size="middle">
        <Alert type="warning" showIcon message="Approximate" description={q.note} />
        <Card size="small" title="GPU hours, rolling 7 days (estimate)">
          <Progress percent={Math.min(100, Math.round(q.pct))} status={q.pct >= 100 ? "exception" : q.pct >= Math.min(...q.warn) ? "active" : "normal"} format={() => `${q.used_h.toFixed(1)} / ${q.limit_h} h`} />
          <Text type="secondary">about {q.left_h.toFixed(1)} h left of your own {q.limit_h} h limit · warnings at {q.warn.join("%, ")}% · {q.projects_counted} project(s) counted. Change the limit in your terminal: kenv quota set limit 30</Text>
        </Card>
        <Chart title="GPU hours per day (UTC), estimate" unit="h" data={q.days} x="day" keys={["hours"]} height={220} />
      </Space>)}</Wait>
  );
}

// ---------------------------------------------------------------- app
function Dashboard() {
  const ov = useApi("overview", {}, true, 15000);
  const rows = ov.data ? ov.data.versions : [];
  const hash = decodeURIComponent(window.location.hash.replace(/^#\/?/, "")).split("/");
  const [tab, setTab] = useState(hash[0] || "versions");
  const [sel, setSel] = useState(hash[1] || null);
  const version = sel && rows.some((r) => r.version === sel) ? sel : ov.data ? (ov.data.only || ov.data.active || (rows.length ? rows[rows.length - 1].version : null)) : null;
  useEffect(() => { window.history.replaceState(null, "", `#/${tab}${version ? "/" + version : ""}`); }, [tab, version]);
  const det = useApi("version", { v: version }, !!version, 15000);
  const pick = useCallback((v) => { setSel(v); setTab("sessions"); }, []);
  if (ov.error) return <div style={{ padding: 24 }}><Fail error={ov.error} /></div>;
  if (!ov.data) return <div style={{ padding: 64, textAlign: "center" }}><Spin size="large" /></div>;
  const o = ov.data;
  const panel = (fn) => <Wait q={det}>{(d) => (<><Fail error={d.core_error} />{fn(d)}</>)}</Wait>;
  const items = [
    { key: "versions", label: "Versions", children: <Versions ov={o} pick={pick} /> },
    { key: "sessions", label: "Sessions", children: panel((d) => <Sessions d={d} />) },
    { key: "runs", label: "Runs & metrics", children: panel((d) => <Runs d={d} rows={rows} />) },
    { key: "resources", label: "Resources", children: panel((d) => <Resources d={d} rows={rows} />) },
    { key: "io", label: "I/O map", children: panel((d) => <IO d={d} />) },
    { key: "deps", label: "Dependencies", children: panel((d) => <Deps d={d} />) },
    { key: "logs", label: "Logs", children: panel((d) => <Logs d={d} live={o.live} />) },
    ...(o.only ? [] : [{ key: "diff", label: "Diff", children: <Diff rows={rows} /> }, { key: "quota", label: "Quota", children: <Quota /> }]),
  ];
  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Layout.Header style={{ display: "flex", alignItems: "center", gap: 16, padding: "0 20px", flexWrap: "wrap", height: "auto", minHeight: 56 }}>
        <Title level={4} style={{ margin: 0, color: "inherit" }}>kenv</Title>
        <Text style={{ color: "inherit", opacity: 0.8 }}>{o.project}</Text>
        <Tag color="cyan">branch {o.branch}</Tag>
        {o.live ? <Tag color="green">session running</Tag> : null}
        {o.only ? <Tag color="gold">single version view</Tag> : null}
        <span style={{ flex: 1 }} />
        {tab !== "versions" && tab !== "diff" && tab !== "quota" ? (
          <Select style={{ minWidth: 220 }} value={version} onChange={setSel} disabled={!!o.only}
            options={[...rows].reverse().map((r) => ({ value: r.version, label: vlabel(r) + (r.active ? " · active" : "") }))} />) : null}
      </Layout.Header>
      <Layout.Content style={{ padding: "12px 20px 40px" }}>
        {rows.length === 0 ? <Empty description="No versions yet: `kenv init` creates v1." /> : <Tabs activeKey={tab} onChange={setTab} items={items} destroyInactiveTabPane />}
      </Layout.Content>
    </Layout>
  );
}

function App() {
  const mq = useMemo(() => window.matchMedia("(prefers-color-scheme: dark)"), []);
  const [dark, setDark] = useState(mq.matches);
  useEffect(() => { const f = (e) => setDark(e.matches); mq.addEventListener("change", f); return () => mq.removeEventListener("change", f); }, [mq]);
  return (
    <ConfigProvider theme={{ algorithm: [dark ? theme.darkAlgorithm : theme.defaultAlgorithm, theme.compactAlgorithm],
      token: { colorPrimary: "#0b7285", borderRadius: 4 }, components: { Layout: { headerBg: dark ? "#0f1c20" : "#0b3b45", headerColor: "#fff" } } }}>
      <Dashboard />
    </ConfigProvider>
  );
}

createRoot(document.getElementById("root")).render(<App />);
