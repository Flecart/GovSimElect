import React, { memo, useEffect, useMemo, useState } from "react";
import {
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { Card } from "./Card";

export interface GovernanceMonitoringRecord {
  propensity: number;
  target: string;
  success: boolean;
  observed: { effort: number; contribution: number; harvest: number } | null;
  punished: boolean;
}

export interface GovernanceRound {
  round: number;
  stock_before: number;
  stock_after_harvest: number;
  stock_after_regeneration: number;
  alpha_before: number;
  alpha_after: number;
  group_norm: string;
  efforts: Record<string, number>;
  contributions: Record<string, number>;
  harvests: Record<string, number>;
  wealth: Record<string, number>;
  monitoring: Record<string, GovernanceMonitoringRecord>;
  punished: string[];
  suspended: string[];
  discussion: { speaker: string; utterance: string }[];
  personal_norms: Record<string, string>;
  proposals: Record<string, string>;
  votes: Record<string, string>;
  collapsed: boolean;
  collapse_reason: string | null;
}

export interface GovernanceExchange {
  sequence: number;
  round: number;
  agent_id: string;
  agent_name: string;
  phase: string;
  messages?: { role: string; content: string }[];
  request: string;
  response: string;
}

export interface GovernanceSummary {
  survival_time?: number;
  collapsed?: boolean;
  collapse_reason?: string | null;
  final_stock?: number;
  final_alpha?: number;
  final_group_norm?: string;
  final_wealth?: Record<string, number>;
  total_contributions?: Record<string, number>;
}

interface GovernanceRunViewProps {
  runName: string;
  rounds: GovernanceRound[];
  messages: GovernanceExchange[];
  summary: GovernanceSummary | null;
}

const PHASE_LABELS: Record<string, string> = {
  harvest_contribution: "Harvest + maintain",
  monitoring: "Monitoring propensity",
  punishment: "Punishment decision",
  discussion: "Public discussion",
  norm_proposal: "Norm proposal",
  community_vote: "Community vote"
};

const AGENT_COLORS: Record<string, string> = {
  John: "#d95d39",
  Kate: "#148f8b",
  Jack: "#d09b2c",
  Emma: "#6c7d3d",
  Luke: "#8d5a97"
};

function numeric(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

const ExchangeCard = memo(function ExchangeCard({
  exchange
}: {
  exchange: GovernanceExchange;
}) {
  const systemMessage = exchange.messages?.find((message) => message.role === "system");
  return (
    <article className="cpr-exchange governance-exchange">
      <header className="cpr-exchange-meta">
        <span
          className="governance-agent-stamp"
          style={{ borderColor: AGENT_COLORS[exchange.agent_name] ?? "#6f766c" }}
        >
          {exchange.agent_name}
        </span>
        <span>Round {exchange.round}</span>
        <span>{PHASE_LABELS[exchange.phase] ?? exchange.phase}</span>
        <span className="cpr-sequence">#{exchange.sequence}</span>
      </header>
      {systemMessage ? (
        <div className="cpr-message governance-message--system">
          <span className="cpr-message-role">System</span>
          <pre>{systemMessage.content}</pre>
        </div>
      ) : null}
      <div className="cpr-message cpr-message--prompt">
        <span className="cpr-message-role">Prompt</span>
        <pre>{exchange.request}</pre>
      </div>
      <div className="cpr-message cpr-message--response">
        <span className="cpr-message-role">Response</span>
        <pre>{exchange.response}</pre>
      </div>
    </article>
  );
});

export const GovernanceRunView: React.FC<GovernanceRunViewProps> = ({
  runName,
  rounds,
  messages,
  summary
}) => {
  const [agentFilter, setAgentFilter] = useState("all");
  const [roundFilter, setRoundFilter] = useState("all");
  const [phaseFilter, setPhaseFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [ledgerRound, setLedgerRound] = useState<number | null>(null);

  const agents = useMemo(() => {
    const values = new Set<string>();
    for (const round of rounds) {
      for (const name of Object.keys(round.efforts ?? {})) values.add(name);
    }
    for (const message of messages) values.add(message.agent_name);
    return Array.from(values);
  }, [messages, rounds]);

  const availableRounds = useMemo(() => {
    const values = new Set<number>();
    for (const round of rounds) values.add(round.round);
    for (const message of messages) values.add(message.round);
    return Array.from(values).sort((a, b) => a - b);
  }, [messages, rounds]);

  useEffect(() => {
    if (ledgerRound === null && rounds.length > 0) {
      setLedgerRound(rounds[rounds.length - 1].round);
    }
  }, [ledgerRound, rounds]);

  useEffect(() => {
    if (agentFilter !== "all" && !agents.includes(agentFilter)) {
      setAgentFilter("all");
    }
  }, [agentFilter, agents]);

  const selectedRound = useMemo(
    () => rounds.find((round) => round.round === ledgerRound) ?? rounds[rounds.length - 1] ?? null,
    [ledgerRound, rounds]
  );

  const filteredMessages = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    const selectedRoundNumber = roundFilter === "all" ? null : Number(roundFilter);
    const selectedPhase = phaseFilter === "all" ? null : phaseFilter;
    const output: GovernanceExchange[] = [];
    for (const message of messages) {
      if (agentFilter !== "all" && message.agent_name !== agentFilter) continue;
      if (selectedRoundNumber !== null && message.round !== selectedRoundNumber) continue;
      if (selectedPhase !== null && message.phase !== selectedPhase) continue;
      if (
        normalizedQuery &&
        !message.request.toLocaleLowerCase().includes(normalizedQuery) &&
        !message.response.toLocaleLowerCase().includes(normalizedQuery)
      ) continue;
      output.push(message);
    }
    return output;
  }, [agentFilter, messages, phaseFilter, query, roundFilter]);

  const telemetry = useMemo(
    () => rounds.map((round) => ({
      round: round.round,
      opening: numeric(round.stock_before),
      afterHarvest: numeric(round.stock_after_harvest),
      regenerated: numeric(round.stock_after_regeneration),
      alpha: numeric(round.alpha_after)
    })),
    [rounds]
  );

  const wealthSeries = useMemo(
    () => rounds.map((round) => {
      const point: Record<string, number> = { round: round.round };
      for (const name of agents) point[name] = numeric(round.wealth?.[name]);
      return point;
    }),
    [agents, rounds]
  );

  const latestRound = rounds[rounds.length - 1] ?? null;
  const collapsed = Boolean(summary?.collapsed ?? latestRound?.collapsed);
  const currentNorm = summary?.final_group_norm ?? latestRound?.group_norm ?? "No norm recorded";

  return (
    <div className="cpr-view governance-view">
      <Card className="cpr-hero governance-hero">
        <Card.Body>
          <div className="cpr-hero-grid">
            <div>
              <div className="cpr-overline">Decentralized commons / live field ledger</div>
              <h1>{runName}</h1>
              <p className="cpr-lede">
                Five fishers · private Mem0 histories · probabilistic monitoring · public norm formation
              </p>
            </div>
            <div className={`cpr-status ${collapsed ? "cpr-status--danger" : ""}`}>
              <span className="cpr-status-light" />
              {collapsed ? `Collapsed · ${summary?.collapse_reason ?? latestRound?.collapse_reason}` : rounds.length ? "Community viable" : "Awaiting first round"}
            </div>
          </div>
          <div className="cpr-metric-ribbon">
            <div><span>Round</span><strong>{latestRound?.round ?? 0}<small>/12</small></strong></div>
            <div><span>Lake stock</span><strong>{numeric(summary?.final_stock, latestRound?.stock_after_regeneration).toFixed(2)}</strong></div>
            <div><span>Efficiency α</span><strong>{numeric(summary?.final_alpha, latestRound?.alpha_after ?? 0.1).toFixed(3)}</strong></div>
            <div><span>Exchanges</span><strong>{messages.length}</strong></div>
          </div>
          <div className="cpr-current-norm">
            <span>Adopted community norm</span>
            <blockquote>{currentNorm}</blockquote>
          </div>
        </Card.Body>
      </Card>

      <div className="cpr-chart-grid">
        <Card className="cpr-panel">
          <Card.Header><h2>Lake and efficiency</h2><p>Stock regenerates by doubling; maintenance shifts next month’s alpha.</p></Card.Header>
          <Card.Body>
            {telemetry.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={telemetry}>
                  <XAxis dataKey="round" tickLine={false} axisLine={false} />
                  <YAxis yAxisId="stock" domain={[0, 100]} tickLine={false} axisLine={false} />
                  <YAxis yAxisId="alpha" orientation="right" domain={[0, 1]} tickLine={false} axisLine={false} />
                  <Tooltip />
                  <ReferenceLine yAxisId="stock" y={5} stroke="#bd4b45" strokeDasharray="4 4" />
                  <Line yAxisId="stock" type="monotone" dataKey="opening" stroke="#426653" dot={false} />
                  <Line yAxisId="stock" type="monotone" dataKey="afterHarvest" stroke="#d95d39" dot={false} />
                  <Line yAxisId="stock" type="monotone" dataKey="regenerated" stroke="#148f8b" strokeWidth={3} dot={false} />
                  <Line yAxisId="alpha" type="stepAfter" dataKey="alpha" stroke="#8d5a97" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : <p className="cpr-empty">Telemetry begins after the first round.</p>}
          </Card.Body>
        </Card>

        <Card className="cpr-panel">
          <Card.Header><h2>Private wealth</h2><p>Harvest minus consumption, contribution, and binary punishment.</p></Card.Header>
          <Card.Body>
            {wealthSeries.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={wealthSeries}>
                  <XAxis dataKey="round" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip />
                  <ReferenceLine y={0} stroke="#bd4b45" strokeDasharray="4 4" />
                  {(agentFilter === "all" ? agents : [agentFilter]).map((name) => (
                    <Line key={name} type="monotone" dataKey={name} stroke={AGENT_COLORS[name] ?? "#6f766c"} dot={false} strokeWidth={agentFilter === "all" ? 1.8 : 3} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : <p className="cpr-empty">Wealth paths begin after the first round.</p>}
          </Card.Body>
        </Card>
      </div>

      {selectedRound ? (
        <Card className="cpr-panel governance-ledger-panel">
          <Card.Header>
            <div className="governance-ledger-title">
              <div><h2>Round ledger</h2><p>Productive choices, private monitoring, and sanctions.</p></div>
              <label>Round<select value={selectedRound.round} onChange={(event) => setLedgerRound(Number(event.target.value))}>{rounds.map((round) => <option key={round.round} value={round.round}>{round.round}</option>)}</select></label>
            </div>
          </Card.Header>
          <Card.Body>
            <div className="governance-ledger">
              <div className="governance-ledger-head"><span>Fisher</span><span>Effort</span><span>Maintain</span><span>Harvest</span><span>Wealth</span><span>Monitor</span><span>Private result</span><span>Status</span></div>
              {agents.map((name) => {
                const monitor = selectedRound.monitoring?.[name];
                const status = selectedRound.punished.includes(name)
                  ? "Punished → suspended"
                  : selectedRound.suspended.includes(name) ? "Suspended" : "Active";
                return (
                  <button key={name} type="button" className="governance-ledger-row" onClick={() => setAgentFilter(name)}>
                    <span><i style={{ background: AGENT_COLORS[name] ?? "#6f766c" }} />{name}</span>
                    <span>{numeric(selectedRound.efforts?.[name]).toFixed(2)}</span>
                    <span>{numeric(selectedRound.contributions?.[name]).toFixed(2)}</span>
                    <span>{numeric(selectedRound.harvests?.[name]).toFixed(2)}</span>
                    <span>{numeric(selectedRound.wealth?.[name]).toFixed(2)}</span>
                    <span>{numeric(monitor?.propensity).toFixed(2)}</span>
                    <span>{monitor?.success ? `${monitor.target}: e ${numeric(monitor.observed?.effort).toFixed(2)}, v ${numeric(monitor.observed?.contribution).toFixed(2)}${monitor.punished ? " · punished" : ""}` : `${monitor?.target ?? "—"}: unseen`}</span>
                    <span>{status}</span>
                  </button>
                );
              })}
            </div>
          </Card.Body>
        </Card>
      ) : null}

      {selectedRound ? (
        <div className="governance-social-grid">
          <Card className="cpr-panel">
            <Card.Header><h2>Public discussion</h2><p>Private monitoring appears here only when a fisher discloses it.</p></Card.Header>
            <Card.Body>
              <ol className="governance-discussion">
                {selectedRound.discussion.map((item, index) => <li key={`${item.speaker}-${index}`}><strong>{item.speaker}</strong><p>{item.utterance}</p></li>)}
                {selectedRound.discussion.length === 0 ? <li>No discussion recorded.</li> : null}
              </ol>
            </Card.Body>
          </Card>
          <Card className="cpr-panel">
            <Card.Header><h2>Proposal and vote</h2><p>Natural-language norms, retained verbatim.</p></Card.Header>
            <Card.Body>
              <div className="governance-proposals">
                {agents.map((name) => <div key={name}><span>{name} · voted for</span><p>{selectedRound.votes?.[name] ?? "No vote"}</p><small>Proposed: {selectedRound.proposals?.[name] ?? "—"}</small></div>)}
              </div>
            </Card.Body>
          </Card>
        </div>
      ) : null}

      <Card className="cpr-panel cpr-transcript-panel">
        <Card.Header>
          <div className="cpr-transcript-title">
            <div><h2>Complete model exchange archive</h2><p>Every prompt and response, filterable by fisher, round, and stage.</p></div>
            <span>{filteredMessages.length} / {messages.length}</span>
          </div>
        </Card.Header>
        <Card.Body>
          <div className="cpr-filters">
            <label>Fisher<select value={agentFilter} onChange={(event) => setAgentFilter(event.target.value)}><option value="all">All fishers</option>{agents.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
            <label>Round<select value={roundFilter} onChange={(event) => setRoundFilter(event.target.value)}><option value="all">All rounds</option>{availableRounds.map((round) => <option key={round} value={round}>Round {round}</option>)}</select></label>
            <label>Stage<select value={phaseFilter} onChange={(event) => setPhaseFilter(event.target.value)}><option value="all">All stages</option>{Object.entries(PHASE_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
            <label className="cpr-search">Search<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="monitor, contribution, norm…" /></label>
          </div>
          <div className="cpr-exchange-list">
            {filteredMessages.map((exchange) => <ExchangeCard key={exchange.sequence} exchange={exchange} />)}
            {filteredMessages.length === 0 ? <p className="cpr-empty">No exchanges match these filters.</p> : null}
          </div>
        </Card.Body>
      </Card>
    </div>
  );
};
