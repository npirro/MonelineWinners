const REQUIRED_COLUMNS = [
  "Game","Team","Opponent","Home","SP_Score","Offense_Score",
  "Bullpen_Score","Lineup_Score","Situational_Score","Notes"
];

const sampleSlate = `Game,Team,Opponent,Home,SP_Score,Offense_Score,Bullpen_Score,Lineup_Score,Situational_Score,Notes
1,PHI,NYM,Yes,88,84,77,86,72,Strong SP and confirmed lineup
1,NYM,PHI,No,71,78,69,75,64,Bullpen concern
2,LAD,SF,Yes,91,89,82,88,78,Best full-game profile
2,SF,LAD,No,66,72,74,69,60,Tough matchup
3,NYY,BOS,No,84,86,76,81,68,Offense advantage
3,BOS,NYY,Yes,73,80,71,77,74,Home field helps
4,SEA,TEX,Yes,79,75,88,76,76,Bullpen edge
4,TEX,SEA,No,74,82,70,79,65,Lineup okay but pitching gap`;

let lastRanked = [];

function parseCSVLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    const next = line[i + 1];
    if (char === '"' && inQuotes && next === '"') {
      current += '"'; i++;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  return result;
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return { rows: [], header: [], errors: ["No usable data found."] };
  const header = parseCSVLine(lines[0]).map(x => x.trim());
  const missing = REQUIRED_COLUMNS.filter(c => !header.includes(c));
  const errors = [];
  if (missing.length) errors.push(`Missing columns: ${missing.join(", ")}`);

  const rows = lines.slice(1).map((line, idx) => {
    const parts = parseCSVLine(line);
    const row = {};
    header.forEach((h, i) => row[h] = parts[i] ?? "");
    row.__row = idx + 2;
    return row;
  });

  for (const row of rows) {
    for (const key of ["SP_Score","Offense_Score","Bullpen_Score","Lineup_Score","Situational_Score"]) {
      const val = Number(row[key]);
      if (Number.isNaN(val) || val < 0 || val > 100) {
        errors.push(`Row ${row.__row}: ${key} must be 0-100.`);
      }
    }
  }
  return { rows, header, errors };
}

function weights() {
  return {
    SP_Score: Number(document.getElementById("wSP").value) || 0,
    Offense_Score: Number(document.getElementById("wOff").value) || 0,
    Bullpen_Score: Number(document.getElementById("wBP").value) || 0,
    Lineup_Score: Number(document.getElementById("wLU").value) || 0,
    Situational_Score: Number(document.getElementById("wSit").value) || 0
  };
}

function scoreRow(row, w) {
  const totalWeight = Object.values(w).reduce((a,b) => a + b, 0) || 1;
  let total = 0;
  for (const key of Object.keys(w)) total += (Number(row[key]) || 0) * w[key];
  return +(total / totalWeight).toFixed(1);
}

function grade(score) {
  if (score >= 85) return ["A", "gradeA"];
  if (score >= 78) return ["B", "gradeB"];
  if (score >= 70) return ["C", "gradeC"];
  return ["Pass", "gradePass"];
}

function strongestMetric(row) {
  const keys = [
    ["SP_Score", "SP"],
    ["Offense_Score", "OFF"],
    ["Bullpen_Score", "BP"],
    ["Lineup_Score", "LU"],
    ["Situational_Score", "SIT"]
  ];
  return keys.reduce((best, k) => (Number(row[k[0]]) || 0) > (Number(row[best[0]]) || 0) ? k : best, keys[0])[1];
}

function profile(row) {
  const flags = [];
  if (Number(row.SP_Score) >= 85) flags.push("SP+");
  if (Number(row.Offense_Score) >= 85) flags.push("OFF+");
  if (Number(row.Bullpen_Score) >= 85) flags.push("BP+");
  if (Number(row.Lineup_Score) >= 85) flags.push("LU+");
  if (Number(row.Situational_Score) >= 80) flags.push("SIT+");
  return flags.length ? flags.join(" / ") : "Balanced";
}

function updateStats(ranked, errors=[]) {
  const aCount = ranked.filter(r => r.WinScore >= 85).length;
  document.getElementById("statTeams").textContent = ranked.length;
  document.getElementById("statA").textContent = aCount;
  document.getElementById("statTop").textContent = ranked.length ? ranked[0].WinScore : "—";
  document.getElementById("statStatus").textContent = errors.length ? "Check Data" : (ranked.length ? "Ready" : "Waiting");
}

function refresh() {
  const raw = document.getElementById("csvInput").value;
  const parsed = parseCSV(raw);
  const validation = document.getElementById("validationBox");

  if (parsed.errors.length) {
    validation.textContent = parsed.errors.slice(0, 6).join(" ");
    validation.className = "validation bad";
  } else {
    validation.textContent = `Data validated. ${parsed.rows.length} teams loaded.`;
    validation.className = "validation good";
  }

  if (!parsed.rows.length || parsed.errors.length) {
    render([]);
    updateStats([], parsed.errors);
    return;
  }

  const w = weights();
  lastRanked = parsed.rows.map(r => ({
    ...r,
    WinScore: scoreRow(r, w),
    Strongest: strongestMetric(r),
    Profile: profile(r)
  })).sort((a,b) => b.WinScore - a.WinScore);

  render(lastRanked);
  updateStats(lastRanked);
}

function render(ranked) {
  const tbody = document.querySelector("#rankTable tbody");
  tbody.innerHTML = "";

  ranked.forEach((r, idx) => {
    const [g, cls] = grade(r.WinScore);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td><b>${r.Team || ""}</b></td>
      <td>${r.Opponent || ""}</td>
      <td>${r.Home || ""}</td>
      <td><b>${r.WinScore}</b></td>
      <td class="${cls}">${g}</td>
      <td>${r.Strongest}</td>
      <td class="profile">${r.Profile}</td>
      <td>${r.Notes || ""}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById("statusPill").textContent = ranked.length
    ? `${ranked.length} teams ranked`
    : "Waiting for data";
}

function exportBoard() {
  if (!lastRanked.length) {
    alert("No ranked board to export yet.");
    return;
  }
  const header = ["Rank","Team","Opponent","Home","WinScore","Grade","StrongestMetric","Profile","Notes"];
  const lines = [header.join(",")];
  lastRanked.forEach((r, idx) => {
    const [g] = grade(r.WinScore);
    const row = [idx+1,r.Team,r.Opponent,r.Home,r.WinScore,g,r.Strongest,r.Profile,r.Notes || ""]
      .map(v => `"${String(v).replaceAll('"','""')}"`);
    lines.push(row.join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `moneyline_winners_board_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

document.getElementById("fileInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById("csvInput").value = await file.text();
  refresh();
});

document.getElementById("loadSampleBtn").addEventListener("click", () => {
  document.getElementById("csvInput").value = sampleSlate;
  refresh();
});
document.getElementById("refreshBtn").addEventListener("click", refresh);
document.getElementById("clearBtn").addEventListener("click", () => {
  document.getElementById("csvInput").value = "";
  lastRanked = [];
  render([]);
  updateStats([]);
  const validation = document.getElementById("validationBox");
  validation.textContent = "No data loaded yet.";
  validation.className = "validation";
});
document.getElementById("exportBtn").addEventListener("click", exportBoard);
document.getElementById("resetWeightsBtn").addEventListener("click", () => {
  document.getElementById("wSP").value = 40;
  document.getElementById("wOff").value = 25;
  document.getElementById("wBP").value = 15;
  document.getElementById("wLU").value = 12;
  document.getElementById("wSit").value = 8;
  if (document.getElementById("csvInput").value.trim()) refresh();
});
