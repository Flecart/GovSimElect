import React, { useMemo } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card } from "./Card";

export interface SummaryGroupRow {
  id?: string;
  group?: string;
  mean_survival_months?: number;
  max_survival_months?: number;
  [key: string]: unknown;
}

export interface ResourceRow {
  x: number;
  round: number;
  [runName: string]: number | null;
}

export interface DashboardViewProps {
  activeGroup: string | null;
  summaryGroups: SummaryGroupRow[];
  resourceInPoolByGroup: Record<string, ResourceRow[]>;
}

interface SurvivalStat {
  survivalMonths: number;
  collapsed: boolean;
}

function computeSurvivalStats(rows: ResourceRow[]): {
  meanSurvival: number;
  maxSurvival: number;
  percentageCollapseByMonth: { month: number; percentage: number }[];
} {
  if (!rows.length) {
    return { meanSurvival: 0, maxSurvival: 0, percentageCollapseByMonth: [] };
  }

  const runNames = Object.keys(rows[0]).filter(
    (k) => k !== "x" && k !== "round"
  ) as string[];
  const stats: SurvivalStat[] = [];

  for (const runName of runNames) {
    let survivalMonths = 12;
    let collapsed = false;

    for (let i = 0; i < rows.length; i += 1) {
      const value = rows[i][runName];
      const v = typeof value === "number" ? value : null;
      if (v === null || v < 5) {
        survivalMonths = Number(rows[i].round ?? 0) + 1;
        collapsed = true;
        break;
      }
    }

    stats.push({ survivalMonths, collapsed });
  }

  const meanSurvival =
    stats.length === 0
      ? 0
      : stats.reduce((acc, s) => acc + s.survivalMonths, 0) / stats.length;
  const maxSurvival = stats.reduce(
    (acc, s) => (s.survivalMonths > acc ? s.survivalMonths : acc),
    0
  );

  const percentageCollapseByMonth: { month: number; percentage: number }[] = [];
  for (let month = 1; month <= 12; month += 1) {
    const collapsedCount = stats.filter(
      (s) => s.collapsed && s.survivalMonths <= month
    ).length;
    const percentage = stats.length ? collapsedCount / stats.length : 0;
    percentageCollapseByMonth.push({ month, percentage });
  }

  return { meanSurvival, maxSurvival, percentageCollapseByMonth };
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  activeGroup,
  summaryGroups,
  resourceInPoolByGroup
}) => {
  const groupRow = useMemo(
    () =>
      summaryGroups.find(
        (row) => row.group === activeGroup || row.id === activeGroup
      ) ?? null,
    [summaryGroups, activeGroup]
  );

  const resourceRows = useMemo(
    () => (activeGroup ? resourceInPoolByGroup[activeGroup] ?? [] : []),
    [resourceInPoolByGroup, activeGroup]
  );

  const meanResourceByMonth = useMemo(() => {
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
      return { month, mean };
    });
  }, [resourceRows]);

  const survivalStats = useMemo(
    () => computeSurvivalStats(resourceRows),
    [resourceRows]
  );

  if (!activeGroup) {
    return (
      <Card>
        <Card.Header>
          <h2>Group dashboard</h2>
        </Card.Header>
        <Card.Body>Select a group to see shared resource statistics.</Card.Body>
      </Card>
    );
  }

  return (
    <div className="run-view">
      <Card className="run-view-section">
        <Card.Header>
          <h2>Group dashboard: {activeGroup}</h2>
        </Card.Header>
        <Card.Body>
          <div className="controls-row">
            <div>
              <div className="stat-label">Mean survival months</div>
              <div className="stat-value">
                {(
                  (groupRow?.mean_survival_months as number | undefined) ??
                  survivalStats.meanSurvival
                ).toFixed(2)}
              </div>
            </div>
            <div>
              <div className="stat-label">Max survival months</div>
              <div className="stat-value">
                {(
                  (groupRow?.max_survival_months as number | undefined) ??
                  survivalStats.maxSurvival
                ).toFixed(0)}
              </div>
            </div>
            <div>
              <div className="stat-label">Collapse at 12 months</div>
              <div className="stat-value">
                {(
                  survivalStats.percentageCollapseByMonth.find(
                    (p) => p.month === 12
                  )?.percentage ?? 0
                ).toLocaleString(undefined, {
                  style: "percent",
                  maximumFractionDigits: 1
                })}
              </div>
            </div>
          </div>
        </Card.Body>
      </Card>

      <div className="run-view-grid">
        <Card className="run-view-section">
          <Card.Header>
            <h3>Shared resource over time</h3>
          </Card.Header>
          <Card.Body>
            {meanResourceByMonth.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={meanResourceByMonth}>
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
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p>No resource time series available for this group.</p>
            )}
          </Card.Body>
        </Card>

        <Card className="run-view-section">
          <Card.Header>
            <h3>Percentage of collapse over time</h3>
          </Card.Header>
          <Card.Body>
            {survivalStats.percentageCollapseByMonth.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={survivalStats.percentageCollapseByMonth}>
                  <XAxis
                    dataKey="month"
                    label={{ value: "Month", position: "insideBottom" }}
                  />
                  <YAxis
                    tickFormatter={(v) =>
                      (v as number).toLocaleString(undefined, {
                        style: "percent",
                        maximumFractionDigits: 0
                      })
                    }
                  />
                  <Tooltip
                    formatter={(v: number) =>
                      v.toLocaleString(undefined, {
                        style: "percent",
                        maximumFractionDigits: 1
                      })
                    }
                  />
                  <Line
                    type="monotone"
                    dataKey="percentage"
                    stroke="#dc2626"
                    dot={false}
                    name="% collapsed"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p>No collapse statistics available for this group.</p>
            )}
          </Card.Body>
        </Card>
      </div>
    </div>
  );
};

