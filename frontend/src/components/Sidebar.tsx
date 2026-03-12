import React from "react";

interface SidebarProps {
  subsets: string[];
  activeSubset: string | null;
  onSubsetChange: (subset: string) => void;
  groups: string[];
  activeGroup: string | null;
  onGroupChange: (group: string) => void;
  runs: { key: string; label: string }[];
  activeRun: string | null;
  selectedRuns: string[];
  onRunChange: (runKey: string) => void;
  onRunToggle: (runKey: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  subsets,
  activeSubset,
  onSubsetChange,
  groups,
  activeGroup,
  onGroupChange,
  runs,
  activeRun,
  selectedRuns,
  onRunChange,
  onRunToggle
}) => {
  return (
    <div className="sidebar">
      <section>
        <h2 className="sidebar-title">Result subsets</h2>
        <ul className="sidebar-list">
          {subsets.map((subset) => (
            <li key={subset}>
              <button
                className={
                  "sidebar-item" +
                  (subset === activeSubset ? " sidebar-item--active" : "")
                }
                onClick={() => onSubsetChange(subset)}
              >
                {subset}
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="sidebar-title">Groups</h2>
        <ul className="sidebar-list">
          {groups.map((group) => (
            <li key={group}>
              <button
                className={
                  "sidebar-item" +
                  (group === activeGroup ? " sidebar-item--active" : "")
                }
                onClick={() => onGroupChange(group)}
              >
                {group}
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="sidebar-title">Runs</h2>
        <ul className="sidebar-list">
          {runs.map((run) => (
            <li key={run.key}>
              <div className="sidebar-run-row">
                <label className="sidebar-run-checkbox">
                  <input
                    type="checkbox"
                    checked={selectedRuns.includes(run.key)}
                    onChange={() => onRunToggle(run.key)}
                  />
                  <span />
                </label>
                <button
                  className={
                    "sidebar-item" + (run.key === activeRun ? " sidebar-item--active" : "")
                  }
                  onClick={() => onRunChange(run.key)}
                >
                  {run.label}
                </button>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
};
