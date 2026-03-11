import React, { useMemo, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card } from "./Card";
import { MarkdownView } from "./MarkdownView";
import { ResourceRow } from "./DashboardView";

export interface RunRow {
  [key: string]: unknown;
}

export interface RunViewProps {
  runName: string | null;
  runData: RunRow[] | null;
  activeGroup: string | null;
  resourceInPoolByGroup: Record<string, ResourceRow[]>;
  selectedRuns: string[];
}

interface ConversationRound {
  round: number;
  utterances: { agent_name: string; utterance: string }[];
}

export const RunView: React.FC<RunViewProps> = ({
  runName,
  runData,
  activeGroup,
  resourceInPoolByGroup,
  selectedRuns
}) => {
  const [activeRound, setActiveRound] = useState<number | null>(null);
  const [activePersona, setActivePersona] = useState<string | null>(null);

  const resourceRows = useMemo(
    () => (activeGroup ? resourceInPoolByGroup[activeGroup] ?? [] : []),
    [resourceInPoolByGroup, activeGroup]
  );

  const resourceChartData = useMemo(() => {
    if (!resourceRows.length) return [];
    return resourceRows.map((row) => {
      const month = Number(row.round ?? 0);
      const entries = Object.keys(row)
        .filter((k) => k !== "x" && k !== "round")
        .map((k) => row[k])
        .filter((v) => typeof v === "number") as number[];
      const mean =
        entries.length === 0
          ? 0
          : entries.reduce((acc, v) => acc + v, 0) / entries.length;
      const point: Record<string, number | string> = { month, mean };
      for (const run of selectedRuns) {
        const v = row[run];
        point[run] = typeof v === "number" ? v : NaN;
      }
      return point;
    });
  }, [resourceRows, selectedRuns]);

  const conversationByRound: ConversationRound[] = useMemo(() => {
    if (!runData) return [];
    const byRound = new Map<number, ConversationRound>();

    for (const row of runData) {
      if (row["action"] !== "utterance") continue;
      const round = Number(row["round"] ?? 0);
      const utterance = String(row["utterance"] ?? "");
      const agentName = String(row["agent_name"] ?? "");

      const bucket =
        byRound.get(round) ??
        {
          round,
          utterances: []
        };
      bucket.utterances.push({ agent_name: agentName, utterance });
      byRound.set(round, bucket);
    }

    return Array.from(byRound.values()).sort((a, b) => a.round - b.round);
  }, [runData]);

  const personas = useMemo(() => {
    if (!runData) return [];
    const ids = new Set<string>();
    for (const row of runData) {
      if (row["agent_id"]) {
        ids.add(String(row["agent_id"]));
      }
    }
    return Array.from(ids.values()).sort();
  }, [runData]);

  const harvestTotalsByPersona = useMemo(() => {
    if (!runData) return {};
    const totals: Record<string, number> = {};
    for (const row of runData) {
      if (row["action"] !== "harvesting") continue;
      const id = String(row["agent_id"] ?? "");
      if (!id || id === "framework") continue;
      const amount = Number(row["resource_collected"] ?? 0);
      totals[id] = (totals[id] ?? 0) + amount;
    }
    return totals;
  }, [runData]);

  const harvestingPrompts = useMemo(() => {
    if (!runData || activeRound == null || !activePersona) return null;
    const rows = runData.filter(
      (row) =>
        row["action"] === "harvesting" &&
        Number(row["round"] ?? 0) === activeRound &&
        String(row["agent_id"] ?? "") === activePersona
    );
    if (!rows.length) return null;
    const html = String(rows[0]["html_interactions"] ?? "");
    return html;
  }, [runData, activeRound, activePersona]);

  const analysisHtml = useMemo(() => {
    if (!runData || activeRound == null) return null;
    const rows = runData.filter(
      (row) =>
        row["action"] === "conversation_resource_limit" &&
        Number(row["round"] ?? 0) === activeRound
    );
    if (!rows.length) return null;
    const html = String(rows[0]["html_interactions"] ?? "");
    return html;
  }, [runData, activeRound]);

  if (!runName || !runData) {
    return (
      <Card>
        <Card.Header>
          <h1>Run details</h1>
        </Card.Header>
        <Card.Body>
          <p>Select a subset, group, and run to inspect agent behaviour.</p>
        </Card.Body>
      </Card>
    );
  }

  return (
    <div className="run-view">
      <Card className="run-view-section">
        <Card.Header>
          <h1>{runName}</h1>
          <p>Shared resource and collapse statistics (current group)</p>
        </Card.Header>
        <Card.Body>
          {resourceChartData.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={resourceChartData}>
                <XAxis
                  dataKey="month"
                  label={{ value: "Month", position: "insideBottom" }}
                />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="mean"
                  stroke="#4f46e5"
                  dot={false}
                  name="Mean resource"
                />
                {selectedRuns.map((run, index) => (
                  <Line
                    key={run}
                    type="monotone"
                    dataKey={run}
                    stroke={index % 2 === 0 ? "#10b981" : "#f97316"}
                    dot={false}
                    name={run}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p>No resource time series available for this group.</p>
          )}
        </Card.Body>
      </Card>

      <div className="run-view-grid">
        <Card className="run-view-section">
          <Card.Header>
            <h2>Conversations by round</h2>
          </Card.Header>
          <Card.Body>
            {Object.keys(harvestTotalsByPersona).length > 0 && (
              <div className="totals-row">
                <strong>Total taken per agent:</strong>
                <ul className="totals-list">
                  {Object.entries(harvestTotalsByPersona).map(([id, total]) => (
                    <li key={id}>
                      <span className="utterance-agent">{id}</span>
                      <span className="utterance-text">{total.toFixed(2)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <div className="round-list">
              {conversationByRound.map((round) => (
                <details
                  key={round.round}
                  open={activeRound === round.round}
                  onClick={() =>
                    setActiveRound((prev) => (prev === round.round ? null : round.round))
                  }
                >
                  <summary>Round {round.round}</summary>
                  <ul className="utterance-list">
                    {round.utterances.map((u, idx) => (
                      <li key={idx}>
                        <span className="utterance-agent">{u.agent_name}</span>
                        <span className="utterance-text">
                          <MarkdownView content={u.utterance} />
                        </span>
                      </li>
                    ))}
                  </ul>
                </details>
              ))}
              {!conversationByRound.length && (
                <p>No utterance logs recorded for this run.</p>
              )}
            </div>
          </Card.Body>
        </Card>

        <Card className="run-view-section">
          <Card.Header>
            <h2>Agent prompts and analysis</h2>
          </Card.Header>
          <Card.Body>
            <div className="controls-row">
              <label>
                Round:
                <select
                  value={activeRound ?? ""}
                  onChange={(e) =>
                    setActiveRound(e.target.value ? Number(e.target.value) : null)
                  }
                >
                  <option value="">(auto from clicks / conversation)</option>
                  {conversationByRound.map((r) => (
                    <option key={r.round} value={r.round}>
                      {r.round}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Persona:
                <select
                  value={activePersona ?? ""}
                  onChange={(e) =>
                    setActivePersona(e.target.value ? e.target.value : null)
                  }
                >
                  <option value="">(all)</option>
                  {personas.map((id) => (
                    <option key={id} value={id}>
                      {id}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="tabs">
              <div className="tabs-row">
                <button
                  type="button"
                  className={
                    "tab" + (harvestingPrompts ? " tab--active" : " tab--inactive")
                  }
                >
                  Harvesting prompts
                </button>
                <button
                  type="button"
                  className={
                    "tab" + (analysisHtml ? " tab--active" : " tab--inactive")
                  }
                >
                  Conversation analysis
                </button>
              </div>
              <div className="tabs-content">
                <section className="tab-panel">
                  <h3>Harvesting prompts</h3>
                  {harvestingPrompts ? (
                    <div
                      className="html-panel"
                      dangerouslySetInnerHTML={{ __html: harvestingPrompts }}
                    />
                  ) : (
                    <p>Select a round and persona to view prompts.</p>
                  )}
                </section>
                <section className="tab-panel">
                  <h3>Conversation analysis</h3>
                  {analysisHtml ? (
                    <div
                      className="html-panel"
                      dangerouslySetInnerHTML={{ __html: analysisHtml }}
                    />
                  ) : (
                    <p>Select a round to view analysis for that conversation.</p>
                  )}
                </section>
              </div>
            </div>
          </Card.Body>
        </Card>
      </div>
    </div>
  );
};

