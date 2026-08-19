import json
import os
import re
import time
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from main import run_pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


# ============================================================================
# OPTIMIZED CACHE SYSTEM
# ============================================================================
class DataCache:
	"""Pre-loads and caches all data structures at startup for optimal performance."""
	
	def __init__(self):
		self.cache_ready = False
		self.communities = []
		self.stats = {
			"total_nodes": 0, "total_edges": 0, "communities": 0, "high_risk": 0,
			"illicit_nodes": 0, "licit_nodes": 0, "unknown_nodes": 0,
			"precision": 0, "recall": 0, "f1": 0, "auc": 0
		}
		self.method_comparison = {}
		self.risk_distribution = {}
		self.graph_cache = {}
		self.explanations_map = {}
		self.load_timestamp = 0
		self.load_error = None
		
	def initialize(self):
		"""Pre-load all data at application startup."""
		try:
			print("[CACHE] Initializing data cache...")
			start_time = time.time()
			
			# Reset all cache containers
			self.communities = []
			self.graph_cache = {}
			self.explanations_map = {}
			self.stats = {}
			self.method_comparison = {}
			self.risk_distribution = {}
			self.load_error = None
			
			# Load explanations first (needed for communities)
			self._load_explanations()
			
			# Load and enrich communities
			self._load_communities()
			
			# Build aggregate stats
			self._build_stats()
			
			# Build method comparison
			self._build_method_comparison()
			
			# Build risk distribution
			self._build_risk_distribution()
			
			# Pre-compute graph data for main communities
			self._precompute_graphs()
			
			self.cache_ready = True
			self.load_timestamp = int(time.time())
			elapsed = time.time() - start_time
			
			print(f"[CACHE] [OK] Cache initialized successfully in {elapsed:.2f}s")
			print(f"[CACHE] [OK] {len(self.communities)} communities loaded")
			print(f"[CACHE] [OK] {len(self.graph_cache)} pre-computed graphs cached")
			return True
			
		except Exception as e:
			self.cache_ready = False
			self.load_error = str(e)
			print(f"[CACHE] [FAIL] ERROR: {e}")
			return False
	
	def _load_explanations(self):
		"""Load community explanations from CSV or text file."""
		explanations_csv = self._read_csv("community_explanations.csv")
		
		if not explanations_csv.empty and "community_id" in explanations_csv.columns:
			for _, row in explanations_csv.iterrows():
				cid = self._safe_int(row.get("community_id"))
				if cid >= 0:
					self.explanations_map[cid] = str(row.get("explanation", ""))
		
		# Fallback to text file
		if not self.explanations_map:
			text_path = os.path.join(OUTPUTS_DIR, "community_explanations.txt")
			if os.path.exists(text_path):
				with open(text_path, "r", encoding="utf-8") as f:
					blocks = [b.strip() for b in f.read().split("\n\n") if b.strip()]
					for block in blocks:
						match = re.search(r"Community\s+(\d+)", block)
						if match:
							self.explanations_map[int(match.group(1))] = block
	
	def _load_communities(self):
		"""Load and enrich communities from CSV."""
		df = self._read_csv("community_stats.csv")
		
		if df.empty:
			print("[CACHE] Note: community_stats.csv not found - this is expected before first pipeline run")
			return
		
		for _, row in df.iterrows():
			cid = self._safe_int(row.get("community_id"))
			if cid < 0:
				continue
			
			# Get or generate explanation
			explanation = self.explanations_map.get(cid, "")
			if not explanation:
				explanation = self._build_explanation_from_row(row)
			
			community = {
				"community_id": cid,
				"total_nodes": self._safe_int(row.get("total_nodes")),
				"illicit_count": self._safe_int(row.get("illicit_count")),
				"licit_count": self._safe_int(row.get("licit_count")),
				"unknown_count": self._safe_int(row.get("unknown_count")),
				"illicit_ratio": self._safe_float(row.get("illicit_ratio")),
				"avg_degree_centrality": self._safe_float(row.get("avg_degree_centrality")),
				"avg_betweenness": self._safe_float(row.get("avg_betweenness")),
				"avg_pagerank": self._safe_float(row.get("avg_pagerank")),
				"avg_clustering": self._safe_float(row.get("avg_clustering")),
				"avg_neighbor_illicit": self._safe_float(row.get("avg_neighbor_illicit")),
				"internal_edge_density": self._safe_float(row.get("internal_edge_density")),
				"temporal_burst_score": self._safe_float(row.get("temporal_burst_score")),
				"risk_score": self._safe_float(row.get("risk_score")),
				"risk_label": str(row.get("risk_label", "LOW")).upper(),
				"attack_type": str(row.get("attack_type", "N/A (Licit Flow)")),
				"explanation": explanation.replace("\n", "<br/>"),
			}
			self.communities.append(community)
		
		# Sort by risk score descending
		self.communities.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
	
	def _build_stats(self):
		"""Build aggregate statistics from dataset and metrics files."""
		dataset_stats = self._read_json("dataset_stats.json")
		metrics = self._read_json("pipeline_metrics.json")
		
		if dataset_stats:
			self.stats = {
				"source_mode": str(dataset_stats.get("source_mode", "elliptic")),
				"total_nodes": self._safe_int(dataset_stats.get("total_nodes")),
				"total_edges": self._safe_int(dataset_stats.get("total_edges")),
				"communities": self._safe_int(dataset_stats.get("communities")),
				"high_risk": self._safe_int(dataset_stats.get("high_risk")),
				"medium_risk": self._safe_int(dataset_stats.get("medium_risk")),
				"low_risk": self._safe_int(dataset_stats.get("low_risk")),
				"illicit_nodes": self._safe_int(dataset_stats.get("illicit_nodes")),
				"licit_nodes": self._safe_int(dataset_stats.get("licit_nodes")),
				"unknown_nodes": self._safe_int(dataset_stats.get("unknown_nodes")),
				"time_steps": self._safe_int(dataset_stats.get("time_steps")),
				"precision": round(self._safe_float(metrics.get("precision")), 4),
				"recall": round(self._safe_float(metrics.get("recall")), 4),
				"f1": round(self._safe_float(metrics.get("f1")), 4),
				"auc": round(self._safe_float(metrics.get("auc")), 4),
			}
		else:
			# Compute from communities
			high = sum(1 for c in self.communities if c["risk_label"] == "HIGH")
			medium = sum(1 for c in self.communities if c["risk_label"] == "MEDIUM")
			low = sum(1 for c in self.communities if c["risk_label"] == "LOW")
			
			self.stats = {
				"source_mode": "elliptic",
				"total_nodes": 0,
				"total_edges": 0,
				"communities": len(self.communities),
				"high_risk": high,
				"medium_risk": medium,
				"low_risk": low,
				"illicit_nodes": 0,
				"licit_nodes": 0,
				"unknown_nodes": 0,
				"time_steps": 0,
				"precision": round(self._safe_float(metrics.get("precision")), 4),
				"recall": round(self._safe_float(metrics.get("recall")), 4),
				"f1": round(self._safe_float(metrics.get("f1")), 4),
				"auc": round(self._safe_float(metrics.get("auc")), 4),
			}
	
	def _build_method_comparison(self):
		"""Build comparison of Louvain vs Label Propagation."""
		summary = self._read_json("pipeline_summary.json")
		
		if summary and "method_comparison" in summary:
			self.method_comparison = summary["method_comparison"]
		else:
			louvain_df = self._read_csv("community_stats.csv")
			lpa_df = self._read_csv("community_stats_lpa.csv")
			
			louvain_count = len(louvain_df)
			louvain_high = int((louvain_df["risk_label"] == "HIGH").sum()) if not louvain_df.empty else 0
			
			lpa_count = len(lpa_df)
			lpa_high = int((lpa_df["risk_label"] == "HIGH").sum()) if not lpa_df.empty else 0
			
			self.method_comparison = {
				"louvain": {"communities": louvain_count, "high_risk": louvain_high, "method": "Louvain"},
				"lpa": {"communities": lpa_count, "high_risk": lpa_high, "method": "Label Propagation"},
				"winner": "Louvain",
				"reason": "Louvain finds cohesive high-density communities better suited for illicit ring detection",
			}
	
	def _build_risk_distribution(self):
		"""Build risk distribution statistics."""
		high = sum(1 for c in self.communities if c["risk_label"] == "HIGH")
		medium = sum(1 for c in self.communities if c["risk_label"] == "MEDIUM")
		low = sum(1 for c in self.communities if c["risk_label"] == "LOW")
		total = max(len(self.communities), 1)
		
		metrics = self._read_json("pipeline_metrics.json")
		
		self.risk_distribution = {
			"high": high,
			"medium": medium,
			"low": low,
			"total": total,
			"high_pct": round(100.0 * high / total, 2),
			"medium_pct": round(100.0 * medium / total, 2),
			"low_pct": round(100.0 * low / total, 2),
			"f1": round(self._safe_float(metrics.get("f1")), 2),
			"auc": round(self._safe_float(metrics.get("auc")), 2),
		}
	
	def _precompute_graphs(self):
		"""Pre-compute and cache graph data for high-risk communities."""
		nodes_df = self._read_csv("nodes_enriched.csv")
		edges_df = self._read_csv("edges.csv")
		
		if nodes_df.empty or edges_df.empty:
			print("[CACHE] Warning: nodes or edges CSV missing - skipping graph pre-compute")
			return
		
		# Cache graph for top 5 communities
		for community in self.communities[:5]:
			cid = community["community_id"]
			try:
				graph_data = self._build_graph_payload(
					community_id=cid,
					nodes_df=nodes_df,
					edges_df=edges_df,
				)
				self.graph_cache[cid] = graph_data
			except Exception as e:
				print(f"[CACHE] Warning: Could not cache graph for community {cid}: {e}")
	
	def _build_graph_payload(
		self,
		community_id: Optional[int] = None,
		nodes_df: Optional[pd.DataFrame] = None,
		edges_df: Optional[pd.DataFrame] = None,
	) -> Dict:
		"""Build graph payload for a community."""
		if nodes_df is None:
			nodes_df = self._read_csv("nodes_enriched.csv")
		if edges_df is None:
			edges_df = self._read_csv("edges.csv")
		
		if nodes_df.empty or edges_df.empty:
			raise ValueError("Node or edge data missing")
		
		# Select community
		selected_community = None
		if community_id:
			for c in self.communities:
				if c["community_id"] == community_id:
					selected_community = c
					break
		
		if not selected_community:
			selected_community = self.communities[0]
		
		selected_id = selected_community["community_id"]
		
		# Filter nodes
		community_nodes = nodes_df[nodes_df["community_id"] == selected_id].copy()
		if community_nodes.empty:
			raise ValueError(f"No nodes found for community {selected_id}")
		
		community_nodes = community_nodes.sort_values("pagerank", ascending=False).head(250)
		node_ids = {str(n) for n in community_nodes["txId"].astype(str)}
		
		# Filter edges
		edges_df["txId1"] = edges_df["txId1"].astype(str)
		edges_df["txId2"] = edges_df["txId2"].astype(str)
		sub_edges = edges_df[
			edges_df["txId1"].isin(node_ids) & edges_df["txId2"].isin(node_ids)
		]
		
		# Compute positions
		graph = nx.Graph()
		graph.add_nodes_from(node_ids)
		graph.add_edges_from(
			sub_edges[["txId1", "txId2"]].itertuples(index=False, name=None)
		)
		
		cache_path = os.path.join(
			OUTPUTS_DIR, f"graph_layout_{selected_id}.json"
		)
		if os.path.exists(cache_path):
			with open(cache_path, "r") as f:
				positions = json.load(f)
		else:
			# Adjust repulsion based on node count to prevent "vague" clumping in small graphs
			k_val = 1.0 / (len(node_ids) ** 0.5) if len(node_ids) > 0 else 0.1
			layout = nx.spring_layout(graph, seed=42, iterations=80, k=k_val * 2.5)
			
			# Increase scale for small graphs to fill the view
			scale_x = 520
			scale_y = 360
			positions = {
				str(n): {"x": float(p[0]) * scale_x, "y": float(p[1]) * scale_y}
				for n, p in layout.items()
			}
			with open(cache_path, "w") as f:
				json.dump(positions, f)
		
		# Compute node risk
		node_risk = self._compute_node_risk(community_nodes)
		community_nodes["node_risk"] = node_risk.values
		
		# Build payloads
		nodes_payload = []
		for _, row in community_nodes.iterrows():
			nid = str(row["txId"])
			pos = positions.get(nid, {"x": 0, "y": 0})
			nodes_payload.append({
				"id": nid,
				"label": str(row.get("label", "unknown")),
				"risk": round(self._safe_float(row.get("node_risk")), 4),
				"pagerank": round(self._safe_float(row.get("pagerank")), 6),
				"betweenness": round(self._safe_float(row.get("betweenness_centrality")), 6),
				"x": round(self._safe_float(pos.get("x")), 3),
				"y": round(self._safe_float(pos.get("y")), 3),
			})
		
		edges_payload = [
			{"source": str(e.txId1), "target": str(e.txId2)}
			for e in sub_edges.itertuples(index=False)
		]
		
		return {
			"nodes": nodes_payload,
			"edges": edges_payload,
			"community_id": selected_id,
			"risk_score": round(selected_community["risk_score"], 4),
			"illicit_ratio": round(selected_community["illicit_ratio"], 4),
		}
	
	def _compute_node_risk(self, nodes_df: pd.DataFrame) -> pd.Series:
		"""Compute bounded node risk from graph features."""
		if nodes_df.empty:
			return pd.Series(dtype=float)
		
		bet = pd.to_numeric(nodes_df.get("betweenness_centrality", 0), errors="coerce").fillna(0)
		pr = pd.to_numeric(nodes_df.get("pagerank", 0), errors="coerce").fillna(0)
		nb = pd.to_numeric(nodes_df.get("neighbor_illicit_ratio", 0), errors="coerce").fillna(0)
		
		bet_max = float(bet.max()) if len(bet) else 0.0001
		pr_max = float(pr.max()) if len(pr) else 0.0001
		
		return (0.55 * nb + 0.25 * (bet / bet_max) + 0.20 * (pr / pr_max)).clip(0, 1)
	
	def _build_explanation_from_row(self, row: Dict) -> str:
		"""Generate explanation when not available."""
		cid = self._safe_int(row.get("community_id"))
		risk = str(row.get("risk_label", "UNKNOWN"))
		illicit_pct = 100 * self._safe_float(row.get("illicit_ratio"))
		
		return (
			f"Community {cid} is flagged {risk} RISK. "
			f"Illicit node ratio: {illicit_pct:.1f}%. "
			f"Average neighbor illicit ratio: {self._safe_float(row.get('avg_neighbor_illicit')):.2f}."
		)
	
	def _read_csv(self, filename: str) -> pd.DataFrame:
		"""Safely read CSV from outputs."""
		path = os.path.join(OUTPUTS_DIR, filename)
		if not os.path.exists(path):
			return pd.DataFrame()
		try:
			return pd.read_csv(path)
		except Exception:
			return pd.DataFrame()
	
	def _read_json(self, filename: str) -> Dict:
		"""Safely read JSON from outputs."""
		path = os.path.join(OUTPUTS_DIR, filename)
		if not os.path.exists(path):
			return {}
		try:
			with open(path, "r") as f:
				return json.load(f)
		except Exception:
			return {}
	
	@staticmethod
	def _safe_float(value, default=0.0) -> float:
		try:
			return float(value)
		except (TypeError, ValueError):
			return default
	
	@staticmethod
	def _safe_int(value, default=0) -> int:
		try:
			return int(value)
		except (TypeError, ValueError):
			return default


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
from pydantic import BaseModel

CONFIG_PATH = os.path.join(BASE_DIR, "data", "config.json")

def load_config() -> Dict:
	if os.path.exists(CONFIG_PATH):
		try:
			with open(CONFIG_PATH, "r") as f:
				return json.load(f)
		except Exception:
			pass
	return {"caching_enabled": True}

def save_config(config: dict):
	os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
	try:
		with open(CONFIG_PATH, "w") as f:
			json.dump(config, f)
	except Exception as e:
		print(f"Failed to write config: {e}")

class ConfigModel(BaseModel):
	caching_enabled: bool

app = FastAPI(
	title="Crypto Community Detection API",
	version="2.0.0",
	description="API for explainable illicit community detection in cryptocurrency networks.",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Initialize global cache
CACHE = DataCache()


@app.on_event("startup")
async def startup_event():
	"""Initialize cache on application startup."""
	os.makedirs(OUTPUTS_DIR, exist_ok=True)
	config = load_config()
	if not config.get("caching_enabled", True):
		print("[CONFIG] Caching is disabled. Purging preloaded output files...")
		_clear_outputs()
		CACHE.cache_ready = False
	else:
		CACHE.initialize()

@app.get("/api/config")
def get_config() -> Dict:
	return load_config()

@app.post("/api/config")
def update_config(config_data: ConfigModel) -> Dict:
	config = {"caching_enabled": config_data.caching_enabled}
	save_config(config)
	if not config_data.caching_enabled:
		_clear_outputs()
		CACHE.communities = []
		CACHE.stats = {
			"total_nodes": 0, "total_edges": 0, "communities": 0, "high_risk": 0,
			"illicit_nodes": 0, "licit_nodes": 0, "unknown_nodes": 0,
			"precision": 0, "recall": 0, "f1": 0, "auc": 0
		}
		CACHE.graph_cache = {}
		CACHE.risk_distribution = {}
		CACHE.method_comparison = {}
		CACHE.cache_ready = False
		print("[CONFIG] Caching disabled: Purged active cache and outputs.")
	else:
		CACHE.initialize()
	return config


def _is_cache_ready():
	return CACHE.cache_ready

# ============================================================================
# API ENDPOINTS
# ============================================================================

from fastapi.staticfiles import StaticFiles

# ...

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend-react", "dist")
assets_dir = os.path.join(FRONTEND_DIR, "assets")
os.makedirs(assets_dir, exist_ok=True)
app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

@app.get("/")
def serve_frontend() -> FileResponse:
	"""Serve the frontend from the API server."""
	index_path = os.path.join(FRONTEND_DIR, "index.html")
	if not os.path.exists(index_path):
		raise HTTPException(status_code=404, detail="React frontend build not found. Run 'npm run build' in frontend-react.")
	return FileResponse(index_path)

@app.get("/{filename}")
def serve_root_files(filename: str) -> FileResponse:
    """Serve other root files like favicon.svg from dist"""
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/api/health")
def health() -> Dict:
	"""Health check with cache status."""
	return {
		"status": "ok",
		"cache_ready": CACHE.cache_ready,
		"timestamp": int(time.time()),
		"cache_load_time": CACHE.load_timestamp,
	}


@app.get("/api/cache-info")
def cache_info() -> Dict:
	"""Get cache initialization details."""
	return {
		"cache_ready": CACHE.cache_ready,
		"communities_cached": len(CACHE.communities),
		"graphs_cached": len(CACHE.graph_cache),
		"load_timestamp": CACHE.load_timestamp,
		"timestamp": int(time.time()),
	}


@app.get("/api/communities")
def get_communities() -> Dict:
	"""Get all communities sorted by risk."""
	return {
		"communities": CACHE.communities,
		"count": len(CACHE.communities),
		"updated_at": CACHE.load_timestamp,
	}


@app.get("/api/community/{community_id}")
def get_community(community_id: int) -> Dict:
	"""Get detailed community metrics and explanation."""
	for row in CACHE.communities:
		if row["community_id"] == community_id:
			return row
	raise HTTPException(status_code=404, detail=f"Community {community_id} not found")


@app.get("/api/stats")
def get_stats() -> Dict:
	"""Get aggregate dataset statistics."""
	return {**CACHE.stats, "updated_at": CACHE.load_timestamp}


@app.get("/api/graph-data")
def get_graph_data(community_id: Optional[int] = Query(None)) -> Dict:
	"""Get graph data for a community."""
	# Check cache first
	if community_id and community_id in CACHE.graph_cache:
		return CACHE.graph_cache[community_id]
	
	# Build on demand
	nodes_df = CACHE._read_csv("nodes_enriched.csv")
	edges_df = CACHE._read_csv("edges.csv")
	
	if nodes_df.empty or edges_df.empty:
		raise HTTPException(status_code=503, detail="Node or edge data unavailable")
	
	try:
		graph_data = CACHE._build_graph_payload(community_id, nodes_df, edges_df)
		return graph_data
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/method-comparison")
def get_method_comparison() -> Dict:
	"""Get Louvain vs Label Propagation comparison."""
	return CACHE.method_comparison


@app.get("/api/risk-distribution")
def get_risk_distribution() -> Dict:
	"""Get risk distribution and metrics."""
	return {**CACHE.risk_distribution, "updated_at": CACHE.load_timestamp}


def _clear_outputs():
	"""Delete all files in the outputs directory to ensure no stale data remains."""
	if os.path.exists(OUTPUTS_DIR):
		import shutil
		for filename in os.listdir(OUTPUTS_DIR):
			file_path = os.path.join(OUTPUTS_DIR, filename)
			try:
				if os.path.isfile(file_path) or os.path.islink(file_path):
					os.unlink(file_path)
				elif os.path.isdir(file_path):
					shutil.rmtree(file_path)
			except Exception as e:
				print(f"Failed to delete {file_path}. Reason: {e}")

@app.post("/api/upload")
async def upload_files_endpoint(files: List[UploadFile] = File(...)):
	"""Endpoint to upload dataset files."""
	data_dir = os.path.join(BASE_DIR, "data")
	os.makedirs(data_dir, exist_ok=True)
	
	try:
		# 1. Clear stale results immediately on new upload
		_clear_outputs()
		# Reset cache state
		CACHE.cache_ready = False
		
		for f in files:
			content = await f.read()
			if f.filename.endswith(".xlsx"):
				# Extract sheets
				import io
				df_dict = pd.read_excel(io.BytesIO(content), sheet_name=None)
				for sheet_name, df in df_dict.items():
					if "feature" in sheet_name.lower():
						df.to_csv(os.path.join(data_dir, "elliptic_txs_features.csv"), index=False, header=False)
					elif "edge" in sheet_name.lower():
						df.to_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"), index=False, header=True)
					elif "class" in sheet_name.lower():
						df.to_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"), index=False, header=True)
			elif f.filename.endswith(".csv"):
				file_path = os.path.join(data_dir, f.filename)
				with open(file_path, "wb") as out:
					out.write(content)
					
		return {"status": "success", "message": "Files uploaded successfully. You may now start the pipeline."}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pipeline-logs")
def get_pipeline_logs() -> Dict:
	"""Fetch current pipeline logs from file."""
	log_path = os.path.join(OUTPUTS_DIR, "pipeline.log")
	if os.path.exists(log_path):
		try:
			with open(log_path, "r", encoding="utf-8") as f:
				lines = f.read().splitlines()
			return {"logs": lines, "status": "running"}
		except Exception as e:
			return {"logs": [f"Error reading logs: {e}"], "status": "error"}
	return {"logs": ["Log buffer empty. Initialize a pipeline run to stream."], "status": "idle"}


@app.post("/api/run-pipeline")
def run_pipeline_endpoint() -> Dict:
	"""Trigger full pipeline and update cache."""
	try:
		summary = run_pipeline(
			data_dir=os.path.join(BASE_DIR, "data"),
			outputs_dir=OUTPUTS_DIR,
		)
		
		# Reinitialize cache with new data
		CACHE.cache_ready = False
		if CACHE.initialize():
			return {
				"status": "success",
				"message": "Pipeline executed and cache refreshed",
				"f1": CACHE.stats.get("f1", 0),
				"summary": summary,
			}
		else:
			raise Exception(CACHE.load_error)
	
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
