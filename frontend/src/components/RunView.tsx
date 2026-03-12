import React, { useEffect, useMemo, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card } from "./Card";
import { MarkdownView } from "./MarkdownView";
import { ResourceRow } from "./DashboardView";

export interface RunRow {
  [key: string]: unknown;
}

export interface ContractingEvent {
  type: string;
  data: Record<string, unknown>;
}

export interface RunViewProps {
  runName: string | null;
  runData: RunRow[] | null;
  contractingEvents: ContractingEvent[];
  activeGroup: string | null;
  resourceInPoolByGroup: Record<string, ResourceRow[]>;
  selectedRuns: string[];
}

interface ConversationRound {
  round: number;
  utterances: { agent_name: string; utterance: string }[];
}

interface ContractTurn {
  agent: string;
  message: string;
  turn: number;
  phase: string;
  html: string;
}

interface ContractSnapshot {
  contract_type?: string;
  content?: string;
  proposer?: string;
  round_created?: number;
  enforcement_status?: string;
  metadata?: Record<string, unknown>;
  conversation_history?: ContractTurn[];
  votes?: Record<string, unknown>;
  agreements?: Record<string, unknown>;
}

interface RoundContractBundle {
  round: number;
  negotiation?: ContractingEvent;
  codedVote?: ContractingEvent;
  enforcement?: ContractingEvent;
  negotiatedContract?: ContractSnapshot | null;
  enforcedContract?: ContractSnapshot | null;
  turns: ContractTurn[];
  toolCalls: ToolCallRecord[];
}

interface ToolCallRecord {
  raw: string;
  parsed: Record<string, unknown> | null;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function getHtml(value: unknown): string | null {
  if (typeof value === "string") {
    return value || null;
  }
  if (Array.isArray(value) && value.length > 0) {
    const first = value[0];
    return typeof first === "string" ? first : null;
  }
  return null;
}

function codeFence(content: string, language: string): string {
  if (!content.trim()) return "";
  return `\`\`\`${language}\n${content.trim()}\n\`\`\``;
}

function formatVoteLabel(value: unknown): string {
  if (typeof value === "boolean") {
    return value ? "YES" : "NO";
  }
  if (typeof value === "string") {
    return value;
  }
  return String(value);
}

function parseToolCalls(text: string): ToolCallRecord[] {
  const matches = text.matchAll(/<TOOL_CALL>([\s\S]*?)<\/TOOL_CALL>/g);
  return Array.from(matches).map((match) => {
    const raw = match[1].trim();
    try {
      return { raw, parsed: JSON.parse(raw) as Record<string, unknown> };
    } catch {
      return { raw, parsed: null };
    }
  });
}

function contractTurnsFrom(contract: ContractSnapshot | null | undefined): ContractTurn[] {
  return asArray<Record<string, unknown>>(contract?.conversation_history).map((turn) => ({
    agent: asString(turn.agent, "Unknown"),
    message: asString(turn.message),
    turn: asNumber(turn.turn),
    phase: asString(turn.phase, "contracting"),
    html: asString(turn.html)
  }));
}

function contractMarkdown(contract: ContractSnapshot | null | undefined): string | null {
  if (!contract) return null;
  const content = asString(contract.content);
  if (!content.trim()) return null;
  if (contract.contract_type === "python_law") {
    return codeFence(content, "python");
  }
  return content;
}

export const RunView: React.FC<RunViewProps> = ({
  runName,
  runData,
  contractingEvents,
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
      const agentName = String(row["agent_name"] ?? row["agent_id"] ?? "");

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
      if (row["agent_id"] && row["agent_id"] !== "framework") {
        ids.add(String(row["agent_id"]));
      }
    }
    return Array.from(ids.values()).sort();
  }, [runData]);

  const harvestRows = useMemo(() => {
    if (!runData) return [];
    return runData
      .filter((row) => row["action"] === "harvesting")
      .map((row) => ({
        round: Number(row["round"] ?? 0),
        agentId: String(row["agent_id"] ?? ""),
        agentName: String(row["agent_name"] ?? row["agent_id"] ?? ""),
        wanted: asNumber(row["wanted_resource"]),
        collected: asNumber(row["resource_collected"]),
        command: asArray<string>(row["commands"])[0] ?? "",
        rewardAdjustment: asNumber(row["contract_reward_adjustment"]),
        html: getHtml(row["html_interactions"])
      }));
  }, [runData]);

  const rounds = useMemo(() => {
    const values = new Set<number>();
    for (const round of conversationByRound) values.add(round.round);
    for (const row of harvestRows) values.add(row.round);
    for (const event of contractingEvents) {
      values.add(asNumber(asRecord(event.data).round));
    }
    return Array.from(values.values()).sort((a, b) => a - b);
  }, [conversationByRound, harvestRows, contractingEvents]);

  useEffect(() => {
    if (activeRound == null && rounds.length) {
      setActiveRound(rounds[0]);
    }
  }, [activeRound, rounds]);

  useEffect(() => {
    if (activePersona == null && personas.length) {
      setActivePersona(personas[0]);
    }
  }, [activePersona, personas]);

  const harvestTotalsByPersona = useMemo(() => {
    const totals: Record<string, number> = {};
    for (const row of harvestRows) {
      if (!row.agentId) continue;
      totals[row.agentId] = (totals[row.agentId] ?? 0) + row.collected;
    }
    return totals;
  }, [harvestRows]);

  const harvestingPrompt = useMemo(() => {
    if (activeRound == null || !activePersona) return null;
    return (
      harvestRows.find(
        (row) => row.round === activeRound && row.agentId === activePersona
      )?.html ?? null
    );
  }, [harvestRows, activeRound, activePersona]);

  const analysisHtml = useMemo(() => {
    if (!runData || activeRound == null) return null;
    const row = runData.find(
      (item) =>
        item["action"] === "conversation_resource_limit" &&
        Number(item["round"] ?? 0) === activeRound
    );
    return row ? getHtml(row["html_interactions"]) : null;
  }, [runData, activeRound]);

  const contractRounds = useMemo<RoundContractBundle[]>(() => {
    const buckets = new Map<number, RoundContractBundle>();
    const getBucket = (round: number) => {
      const existing = buckets.get(round);
      if (existing) return existing;
      const created: RoundContractBundle = {
        round,
        turns: [],
        toolCalls: []
      };
      buckets.set(round, created);
      return created;
    };

    for (const event of contractingEvents) {
      const data = asRecord(event.data);
      const round = asNumber(data.round, -1);
      const bucket = getBucket(round);
      const contract = asRecord(data.contract);
      const snapshot =
        Object.keys(contract).length === 0 ? null : (contract as ContractSnapshot);
      if (snapshot) {
        if (event.type === "negotiation") {
          bucket.negotiatedContract = snapshot;
        } else if (event.type === "enforcement") {
          bucket.enforcedContract = snapshot;
        }
      }
      if (event.type === "negotiation") {
        bucket.negotiation = event;
      } else if (event.type === "coded_vote") {
        bucket.codedVote = event;
      } else if (event.type === "enforcement") {
        bucket.enforcement = event;
      }
    }

    for (const bucket of buckets.values()) {
      const sourceContract =
        bucket.negotiatedContract ??
        bucket.enforcedContract ??
        null;
      const turns = contractTurnsFrom(sourceContract);
      bucket.turns = turns;
      bucket.toolCalls = turns.flatMap((turn) => parseToolCalls(turn.message));
    }

    return Array.from(buckets.values()).sort((a, b) => a.round - b.round);
  }, [contractingEvents]);

  const selectedContractRound = useMemo(
    () => contractRounds.find((bundle) => bundle.round === activeRound) ?? null,
    [contractRounds, activeRound]
  );

  const latestContract = useMemo(() => {
    for (let index = contractRounds.length - 1; index >= 0; index -= 1) {
      if (contractRounds[index].negotiatedContract) {
        return contractRounds[index].negotiatedContract ?? null;
      }
      if (contractRounds[index].enforcedContract) {
        return contractRounds[index].enforcedContract ?? null;
      }
    }
    return null;
  }, [contractRounds]);

  const selectedRoundHarvests = useMemo(() => {
    if (activeRound == null) return [];
    return harvestRows.filter((row) => row.round === activeRound);
  }, [harvestRows, activeRound]);

  const toolCallCount = useMemo(
    () => contractRounds.reduce((acc, bundle) => acc + bundle.toolCalls.length, 0),
    [contractRounds]
  );

  const codedVotePasses = useMemo(
    () =>
      contractRounds.filter(
        (bundle) => asRecord(bundle.codedVote?.data).passed === true
      ).length,
    [contractRounds]
  );

  const totalAdjustments = useMemo(
    () =>
      harvestRows.reduce((acc, row) => acc + Math.abs(row.rewardAdjustment), 0),
    [harvestRows]
  );

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

  const latestNlLaw = latestContract
    ? asString(asRecord(latestContract.metadata).nl_contract, latestContract.content ?? "")
    : "";
  const latestCodeLaw = latestContract?.contract_type === "python_law"
    ? asString(latestContract.content)
    : "";

  return (
    <div className="run-view">
      <Card className="run-view-section run-view-hero">
        <Card.Header>
          <div className="hero-header">
            <div>
              <h1>{runName}</h1>
              <p>
                Active contracting state, code-law trace, tool use, and concurrent
                fishing execution for this run.
              </p>
            </div>
            <div className="metric-strip">
              <div className="metric-pill">
                <span className="metric-label">Contract rounds</span>
                <strong>{contractRounds.length}</strong>
              </div>
              <div className="metric-pill">
                <span className="metric-label">Tool calls</span>
                <strong>{toolCallCount}</strong>
              </div>
              <div className="metric-pill">
                <span className="metric-label">Code votes passed</span>
                <strong>{codedVotePasses}</strong>
              </div>
              <div className="metric-pill">
                <span className="metric-label">|Reward adj.|</span>
                <strong>{totalAdjustments.toFixed(2)}</strong>
              </div>
            </div>
          </div>
        </Card.Header>
        <Card.Body>
          <div className="hero-grid">
            <div className="hero-panel">
              <div className="section-kicker">Current English law</div>
              {latestNlLaw ? (
                <MarkdownView content={latestNlLaw} />
              ) : (
                <p>No formal law recorded yet.</p>
              )}
            </div>
            <div className="hero-panel">
              <div className="section-kicker">Current code law</div>
              {latestCodeLaw ? (
                <MarkdownView content={codeFence(latestCodeLaw, "python")} />
              ) : (
                <p>No Python-law code is active for this run.</p>
              )}
            </div>
          </div>
        </Card.Body>
      </Card>

      <Card className="run-view-section">
        <Card.Header>
          <h2>Shared resource and collapse statistics</h2>
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
                  stroke="#d97706"
                  dot={false}
                  name="Mean resource"
                />
                {selectedRuns.map((run, index) => (
                  <Line
                    key={run}
                    type="monotone"
                    dataKey={run}
                    stroke={["#0f766e", "#2563eb", "#be123c", "#7c3aed"][index % 4]}
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
            <h2>Contract timeline</h2>
          </Card.Header>
          <Card.Body>
            {contractRounds.length ? (
              <div className="contract-timeline">
                {contractRounds.map((bundle) => {
                  const negotiationData = asRecord(bundle.negotiation?.data);
                  const enforcementData = asRecord(bundle.enforcement?.data);
                  const codedVoteData = asRecord(bundle.codedVote?.data);
                  const negotiatedContract = bundle.negotiatedContract ?? null;
                  const enforcedContract = bundle.enforcedContract ?? null;
                  const result = asRecord(enforcementData.result);
                  const votes = asRecord(codedVoteData.votes);
                  const executionLog = asArray<string>(result.execution_log);
                  return (
                    <details
                      key={bundle.round}
                      className="contract-round"
                      open={activeRound === bundle.round}
                      onClick={() =>
                        setActiveRound((prev) => (prev === bundle.round ? null : bundle.round))
                      }
                    >
                      <summary>
                        <span>Round {bundle.round}</span>
                        <span className="contract-summary-chip">
                          {asString(negotiationData.summary, "Contract activity")}
                        </span>
                      </summary>

                      {bundle.toolCalls.length > 0 && (
                        <div className="contract-block">
                          <div className="section-kicker">Tool calls</div>
                          <div className="tool-grid">
                            {bundle.toolCalls.map((toolCall, index) => (
                              <div key={`${bundle.round}-${index}`} className="tool-card">
                                <strong>
                                  {asString(toolCall.parsed?.tool, "tool_call")}
                                </strong>
                                <MarkdownView
                                  content={
                                    toolCall.parsed
                                      ? codeFence(
                                          JSON.stringify(toolCall.parsed, null, 2),
                                          "json"
                                        )
                                      : codeFence(toolCall.raw, "json")
                                  }
                                />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {enforcedContract && (
                        <div className="contract-block">
                          <div className="section-kicker">Law used for enforcement this round</div>
                          <div className="badge-row">
                            <span className="status-badge">
                              {enforcedContract.contract_type ?? "unknown"}
                            </span>
                            <span className="status-badge">
                              proposer: {enforcedContract.proposer ?? "n/a"}
                            </span>
                            <span className="status-badge">
                              created: {enforcedContract.round_created ?? "n/a"}
                            </span>
                          </div>
                          <div className="contract-law-grid">
                            <div>
                              <div className="section-kicker">Natural-language law</div>
                              <MarkdownView
                                content={
                                  asString(asRecord(enforcedContract.metadata).nl_contract) ||
                                  "No natural-language law recorded."
                                }
                              />
                            </div>
                            <div>
                              <div className="section-kicker">Rendered contract code</div>
                              {contractMarkdown(enforcedContract) ? (
                                <MarkdownView content={contractMarkdown(enforcedContract) ?? ""} />
                              ) : (
                                <p>No code attached to this contract.</p>
                              )}
                            </div>
                          </div>
                        </div>
                      )}

                      {negotiatedContract && (
                        <div className="contract-block">
                          <div className="section-kicker">Law adopted at end of round</div>
                          <div className="badge-row">
                            <span className="status-badge">
                              {negotiatedContract.contract_type ?? "unknown"}
                            </span>
                            <span className="status-badge">
                              proposer: {negotiatedContract.proposer ?? "n/a"}
                            </span>
                            <span className="status-badge">
                              created: {negotiatedContract.round_created ?? "n/a"}
                            </span>
                          </div>
                          <div className="contract-law-grid">
                            <div>
                              <div className="section-kicker">Negotiated English law</div>
                              <MarkdownView
                                content={
                                  asString(asRecord(negotiatedContract.metadata).nl_contract) ||
                                  "No natural-language law recorded."
                                }
                              />
                            </div>
                            <div>
                              <div className="section-kicker">Negotiated contract code</div>
                              {contractMarkdown(negotiatedContract) ? (
                                <MarkdownView content={contractMarkdown(negotiatedContract) ?? ""} />
                              ) : (
                                <p>No code attached to this contract.</p>
                              )}
                            </div>
                          </div>
                        </div>
                      )}

                      {Object.keys(votes).length > 0 && (
                        <div className="contract-block">
                          <div className="section-kicker">Coded-contract vote</div>
                          <div className="badge-row">
                            <span className="status-badge">
                              passed: {codedVoteData.passed === true ? "yes" : "no"}
                            </span>
                            <span className="status-badge">
                              retry: {codedVoteData.retry_attempt ?? 0}
                            </span>
                          </div>
                          <ul className="kv-list">
                            {Object.entries(votes).map(([agent, value]) => (
                              <li key={agent}>
                                <span>{agent}</span>
                                <strong>{formatVoteLabel(value)}</strong>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {bundle.turns.length > 0 && (
                        <div className="contract-block">
                          <div className="section-kicker">Contracting dialogue</div>
                          <div className="contract-turn-list">
                            {bundle.turns.map((turn, index) => (
                              <div key={`${bundle.round}-turn-${index}`} className="contract-turn">
                                <div className="contract-turn-meta">
                                  <span>{turn.agent}</span>
                                  <span>{turn.phase}</span>
                                </div>
                                <MarkdownView content={turn.message} />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {executionLog.length > 0 && (
                        <div className="contract-block">
                          <div className="section-kicker">Enforcement execution log</div>
                          <ul className="execution-log">
                            {executionLog.map((item, index) => (
                              <li key={`${bundle.round}-exec-${index}`}>
                                <code>{item}</code>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </details>
                  );
                })}
              </div>
            ) : (
              <p>No contracting events were logged for this run.</p>
            )}
          </Card.Body>
        </Card>

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
            <h2>Round execution</h2>
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
                  <option value="">(select round)</option>
                  {rounds.map((round) => (
                    <option key={round} value={round}>
                      {round}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {selectedRoundHarvests.length ? (
              <div className="execution-table">
                <div className="execution-table-header">
                  <span>Agent</span>
                  <span>Command</span>
                  <span>Wanted</span>
                  <span>Collected</span>
                  <span>Reward adj.</span>
                </div>
                {selectedRoundHarvests.map((row) => (
                  <div key={`${row.round}-${row.agentId}`} className="execution-table-row">
                    <span>{row.agentName}</span>
                    <code>{row.command || "n/a"}</code>
                    <span>{row.wanted.toFixed(2)}</span>
                    <span>{row.collected.toFixed(2)}</span>
                    <span>{row.rewardAdjustment.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p>No harvesting execution recorded for the selected round.</p>
            )}

            {selectedContractRound?.enforcement && (
              <div className="contract-block">
                <div className="section-kicker">Enforcement state</div>
                <MarkdownView
                  content={codeFence(
                    JSON.stringify(asRecord(asRecord(selectedContractRound.enforcement.data).result), null, 2),
                    "json"
                  )}
                />
              </div>
            )}
          </Card.Body>
        </Card>

        <Card className="run-view-section">
          <Card.Header>
            <h2>Prompt inspector</h2>
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
                  <option value="">(select round)</option>
                  {rounds.map((round) => (
                    <option key={round} value={round}>
                      {round}
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
                  <option value="">(select persona)</option>
                  {personas.map((id) => (
                    <option key={id} value={id}>
                      {id}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="inspector-grid">
              <section className="tab-panel">
                <h3>Harvesting prompt</h3>
                {harvestingPrompt ? (
                  <div
                    className="html-panel"
                    dangerouslySetInnerHTML={{ __html: harvestingPrompt }}
                  />
                ) : (
                  <p>Select a round and persona to view the harvest prompt trace.</p>
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
                  <p>Select a round to view the framework negotiation analysis.</p>
                )}
              </section>
            </div>
          </Card.Body>
        </Card>
      </div>
    </div>
  );
};
