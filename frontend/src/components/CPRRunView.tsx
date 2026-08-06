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

export interface CPRRound {
  round: number;
  stock_before: number;
  stock_after_harvest: number;
  stock_after_regeneration: number;
  group_norm: string;
  efforts: Record<string, number>;
  harvests: Record<string, number>;
  wealth: Record<string, number>;
  punishments: Record<string, number | null>;
  proposals: Record<string, string>;
  votes: Record<string, string>;
  collapsed: boolean;
  efficiency: number;
}

export interface CPRExchange {
  sequence: number;
  round: number;
  agent_id: number;
  phase: string;
  request: string;
  response: string;
}

export interface CPRSummary {
  survival_time?: number;
  collapsed?: boolean;
  mean_efficiency?: number;
  final_stock?: number;
  final_group_norm?: string;
}

interface CPRRunViewProps {
  runName: string;
  rounds: CPRRound[];
  messages: CPRExchange[];
  summary: CPRSummary | null;
}

const PHASE_LABELS: Record<string, string> = {
  effort: "Harvest effort",
  punishment: "Sanction",
  norm_proposal: "Norm proposal",
  community_vote: "Community vote"
};

const AGENT_COLORS = [
  "#e96d38",
  "#177e89",
  "#f2b134",
  "#5c6f3e",
  "#be4b5e",
  "#3767a6",
  "#9a5b9e",
  "#7d5a3c",
  "#2f8f68",
  "#d1495b"
];

function numeric(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

const ExchangeCard = memo(function ExchangeCard({
  exchange
}: {
  exchange: CPRExchange;
}) {
  return (
    <article className="cpr-exchange">
      <header className="cpr-exchange-meta">
        <span className="cpr-agent-stamp">V{exchange.agent_id}</span>
        <span>Round {exchange.round}</span>
        <span>{PHASE_LABELS[exchange.phase] ?? exchange.phase}</span>
        <span className="cpr-sequence">#{exchange.sequence}</span>
      </header>
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

export const CPRRunView: React.FC<CPRRunViewProps> = ({
  runName,
  rounds,
  messages,
  summary
}) => {
  const [agentFilter, setAgentFilter] = useState("all");
  const [roundFilter, setRoundFilter] = useState("all");
  const [phaseFilter, setPhaseFilter] = useState("all");
  const [query, setQuery] = useState("");

  const agents = useMemo(() => {
    const values = new Set<number>();
    for (const message of messages) values.add(message.agent_id);
    for (const round of rounds) {
      for (const agentId of Object.keys(round.efforts ?? {})) {
        values.add(Number(agentId));
      }
    }
    return Array.from(values).sort((a, b) => a - b);
  }, [messages, rounds]);

  const availableRounds = useMemo(() => {
    const values = new Set<number>();
    for (const round of rounds) values.add(round.round);
    for (const message of messages) values.add(message.round);
    return Array.from(values).sort((a, b) => a - b);
  }, [messages, rounds]);

  useEffect(() => {
    if (agentFilter !== "all" && !agents.includes(Number(agentFilter))) {
      setAgentFilter("all");
    }
  }, [agentFilter, agents]);

  const filteredMessages = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    const selectedAgent = agentFilter === "all" ? null : Number(agentFilter);
    const selectedRound = roundFilter === "all" ? null : Number(roundFilter);
    const selectedPhase = phaseFilter === "all" ? null : phaseFilter;
    const output: CPRExchange[] = [];

    for (const message of messages) {
      if (selectedAgent !== null && message.agent_id !== selectedAgent) continue;
      if (selectedRound !== null && message.round !== selectedRound) continue;
      if (selectedPhase !== null && message.phase !== selectedPhase) continue;
      if (
        normalizedQuery &&
        !message.request.toLocaleLowerCase().includes(normalizedQuery) &&
        !message.response.toLocaleLowerCase().includes(normalizedQuery)
      ) {
        continue;
      }
      output.push(message);
    }
    return output;
  }, [agentFilter, messages, phaseFilter, query, roundFilter]);

  const stockSeries = useMemo(
    () =>
      rounds.map((round) => ({
        round: round.round,
        opening: numeric(round.stock_before),
        afterHarvest: numeric(round.stock_after_harvest),
        regenerated: numeric(round.stock_after_regeneration)
      })),
    [rounds]
  );

  const wealthSeries = useMemo(
    () =>
      rounds.map((round) => {
        const point: Record<string, number> = { round: round.round };
        for (const agent of agents) {
          point[`V${agent}`] = numeric(round.wealth?.[String(agent)]);
        }
        return point;
      }),
    [agents, rounds]
  );

  const visibleWealthAgents =
    agentFilter === "all" ? agents : [Number(agentFilter)];
  const latestRound = rounds.length ? rounds[rounds.length - 1] : null;
  const currentNorm =
    summary?.final_group_norm ?? latestRound?.group_norm ?? "No norm recorded yet";
  const currentStock = numeric(
    summary?.final_stock,
    latestRound?.stock_after_regeneration ?? 100
  );
  const collapsed = Boolean(summary?.collapsed ?? latestRound?.collapsed);

  return (
    <div className="cpr-view">
      <Card className="cpr-hero">
        <Card.Body>
          <div className="cpr-hero-grid">
            <div>
              <div className="cpr-overline">Field station / live experiment</div>
              <h1>{runName}</h1>
              <p className="cpr-lede">
                Harsh lake · selfish priors · full social learning · private Mem0 histories
              </p>
            </div>
            <div className={`cpr-status ${collapsed ? "cpr-status--danger" : ""}`}>
              <span className="cpr-status-light" />
              {collapsed ? "Collapsed" : rounds.length ? "Running / viable" : "Awaiting first round"}
            </div>
          </div>
          <div className="cpr-metric-ribbon">
            <div><span>Round</span><strong>{latestRound?.round ?? 0}<small>/50</small></strong></div>
            <div><span>Lake stock</span><strong>{currentStock.toFixed(2)}</strong></div>
            <div><span>Mean efficiency</span><strong>{numeric(summary?.mean_efficiency, latestRound?.efficiency).toFixed(2)}</strong></div>
            <div><span>Exchanges</span><strong>{messages.length}</strong></div>
          </div>
          <div className="cpr-current-norm">
            <span>Current community norm</span>
            <blockquote>{currentNorm}</blockquote>
          </div>
        </Card.Body>
      </Card>

      <div className="cpr-chart-grid">
        <Card className="cpr-panel">
          <Card.Header><h2>Lake pulse</h2><p>Opening, harvested, and regenerated stock.</p></Card.Header>
          <Card.Body>
            {stockSeries.length ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={stockSeries}>
                  <XAxis dataKey="round" tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 300]} tickLine={false} axisLine={false} />
                  <Tooltip />
                  <ReferenceLine y={5} stroke="#be4b5e" strokeDasharray="4 4" label="collapse" />
                  <Line type="monotone" dataKey="opening" stroke="#355f52" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="afterHarvest" stroke="#e96d38" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="regenerated" stroke="#177e89" dot={false} strokeWidth={3} />
                </LineChart>
              </ResponsiveContainer>
            ) : <p className="cpr-empty">Stock telemetry will appear after round one.</p>}
          </Card.Body>
        </Card>

        <Card className="cpr-panel">
          <Card.Header><h2>Private fortunes</h2><p>Accumulated payoff by villager.</p></Card.Header>
          <Card.Body>
            {wealthSeries.length ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={wealthSeries}>
                  <XAxis dataKey="round" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip />
                  <ReferenceLine y={0} stroke="#be4b5e" strokeDasharray="4 4" />
                  {visibleWealthAgents.map((agent) => (
                    <Line
                      key={agent}
                      type="monotone"
                      dataKey={`V${agent}`}
                      stroke={AGENT_COLORS[agent % AGENT_COLORS.length]}
                      dot={false}
                      strokeWidth={agentFilter === "all" ? 1.7 : 3}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : <p className="cpr-empty">Wealth telemetry will appear after round one.</p>}
          </Card.Body>
        </Card>
      </div>

      {latestRound ? (
        <Card className="cpr-panel">
          <Card.Header><h2>Latest village ledger</h2><p>Round {latestRound.round}, after sanctions and regeneration.</p></Card.Header>
          <Card.Body>
            <div className="cpr-ledger">
              <div className="cpr-ledger-head"><span>Villager</span><span>Effort</span><span>Harvest</span><span>Wealth</span><span>Sanctioned</span></div>
              {agents.map((agent) => {
                const target = latestRound.punishments?.[String(agent)];
                return (
                  <button key={agent} type="button" className="cpr-ledger-row" onClick={() => setAgentFilter(String(agent))}>
                    <span><i style={{ background: AGENT_COLORS[agent % AGENT_COLORS.length] }} />V{agent}</span>
                    <span>{numeric(latestRound.efforts?.[String(agent)]).toFixed(2)}</span>
                    <span>{numeric(latestRound.harvests?.[String(agent)]).toFixed(2)}</span>
                    <span>{numeric(latestRound.wealth?.[String(agent)]).toFixed(2)}</span>
                    <span>{target == null ? "—" : `V${target}`}</span>
                  </button>
                );
              })}
            </div>
          </Card.Body>
        </Card>
      ) : null}

      <Card className="cpr-panel cpr-transcript-panel">
        <Card.Header>
          <div className="cpr-transcript-title">
            <div><h2>Complete exchange archive</h2><p>Every prompt and response, including each villager’s retrieved private history.</p></div>
            <span>{filteredMessages.length} / {messages.length}</span>
          </div>
        </Card.Header>
        <Card.Body>
          <div className="cpr-filters">
            <label>Villager<select value={agentFilter} onChange={(event) => setAgentFilter(event.target.value)}><option value="all">All villagers</option>{agents.map((agent) => <option key={agent} value={agent}>Villager {agent}</option>)}</select></label>
            <label>Round<select value={roundFilter} onChange={(event) => setRoundFilter(event.target.value)}><option value="all">All rounds</option>{availableRounds.map((round) => <option key={round} value={round}>Round {round}</option>)}</select></label>
            <label>Phase<select value={phaseFilter} onChange={(event) => setPhaseFilter(event.target.value)}><option value="all">All phases</option>{Object.entries(PHASE_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
            <label className="cpr-search">Search<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="norm, payoff, private history…" /></label>
          </div>
          <div className="cpr-exchange-list">
            {filteredMessages.map((exchange) => <ExchangeCard key={exchange.sequence} exchange={exchange} />)}
            {!filteredMessages.length ? <p className="cpr-empty">No exchanges match these filters yet.</p> : null}
          </div>
        </Card.Body>
      </Card>
    </div>
  );
};
