-- Holdout inference eval (run after inference.py).
-- Source: sandbox.group_data_science.cas_search_reranking_lgbm_inferences

CREATE OR REPLACE TEMP TABLE search_inferences AS
SELECT
  fanduel_user_id AS user_id,
  game_id,
  session_id,
  event_date,
  final_similarity_score AS final_score,
  click_proba
FROM sandbox.group_data_science.cas_search_reranking_lgbm_inferences;

-- ---------------------------------------------------------------------------
-- click_proba distribution (0.05-wide buckets on [0, 1]; use *10 and /10.0 + 0.1 for 0.1 bins)
-- ---------------------------------------------------------------------------
SELECT
  CONCAT(
    CAST(FLOOR(click_proba * 20) / 20.0 AS DECIMAL(4, 2)),
    '-',
    CAST(LEAST(FLOOR(click_proba * 20) / 20.0 + 0.05, 1.0) AS DECIMAL(4, 2))
  ) AS click_proba_bucket,
  COUNT(*) AS row_count
FROM search_inferences
WHERE click_proba IS NOT NULL
GROUP BY 1
ORDER BY MIN(click_proba);

-- ---------------------------------------------------------------------------
-- Enriched rows: normalized scores + rerank metrics (full inference table)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE TEMP VIEW inference_rerank_metrics AS
WITH src AS (
  SELECT *
  FROM sandbox.group_data_science.cas_search_reranking_lgbm_inferences
  WHERE click_proba IS NOT NULL
    AND final_similarity_score IS NOT NULL
),
bounds AS (
  SELECT
    MIN(final_similarity_score) AS fs_min,
    MAX(final_similarity_score) AS fs_max,
    MIN(click_proba) AS cp_min,
    MAX(click_proba) AS cp_max
  FROM src
),
normed AS (
  SELECT
    s.*,
    (s.final_similarity_score - b.fs_min) / NULLIF(b.fs_max - b.fs_min, 0) AS norm_final_score,
    (s.click_proba - b.cp_min) / NULLIF(b.cp_max - b.cp_min, 0) AS norm_click_proba
  FROM src s
  CROSS JOIN bounds b
),
ranked AS (
  SELECT
    *,
    norm_final_score - norm_click_proba AS reranked_game_position_delta,
    ROW_NUMBER() OVER (
      PARTITION BY fanduel_user_id, session_id, search_keyword
      ORDER BY click_proba DESC
    ) AS reranked_game_position
  FROM normed
)
SELECT
  *,
  ABS(game_position - reranked_game_position) AS game_position_abs_delta
FROM ranked;

-- reranked_game_position_delta distribution (0.05-wide buckets)
SELECT
  CONCAT(
    CAST(FLOOR(reranked_game_position_delta * 20) / 20.0 AS DECIMAL(4, 2)),
    '-',
    CAST(FLOOR(reranked_game_position_delta * 20) / 20.0 + 0.05 AS DECIMAL(4, 2))
  ) AS reranked_game_position_delta_bucket,
  COUNT(*) AS row_count
FROM inference_rerank_metrics
GROUP BY 1
ORDER BY MIN(reranked_game_position_delta);

-- ---------------------------------------------------------------------------
-- Sanity check: max logged slot == max reranked slot per search context
-- (should return zero rows if every group has the same top slot index)
-- ---------------------------------------------------------------------------
SELECT
  fanduel_user_id,
  session_id,
  search_keyword,
  MAX(game_position) AS max_logged_game_position,
  MAX(reranked_game_position) AS max_reranked_game_position,
  COUNT(*) AS games_in_context
FROM inference_rerank_metrics
GROUP BY fanduel_user_id, session_id, search_keyword
HAVING MAX(game_position) IS DISTINCT FROM MAX(reranked_game_position);

-- Summary: how many contexts fail the max-position check
SELECT
  COUNT(*) AS contexts_with_mismatched_max_position
FROM (
  SELECT
    fanduel_user_id,
    session_id,
    search_keyword
  FROM inference_rerank_metrics
  GROUP BY fanduel_user_id, session_id, search_keyword
  HAVING MAX(game_position) IS DISTINCT FROM MAX(reranked_game_position)
);

-- ---------------------------------------------------------------------------
-- Volume by reranked_game_position_delta (0.05-wide buckets; same as distribution above)
-- ---------------------------------------------------------------------------
SELECT
  reranked_game_position_delta_bucket,
  COUNT(*) AS game_count
FROM (
  SELECT
    CONCAT(
      CAST(FLOOR(reranked_game_position_delta * 20) / 20.0 AS DECIMAL(4, 2)),
      '-',
      CAST(FLOOR(reranked_game_position_delta * 20) / 20.0 + 0.05 AS DECIMAL(4, 2))
    ) AS reranked_game_position_delta_bucket,
    FLOOR(reranked_game_position_delta * 20) AS sort_key
  FROM inference_rerank_metrics
  WHERE reranked_game_position_delta IS NOT NULL
) t
GROUP BY reranked_game_position_delta_bucket, sort_key
ORDER BY sort_key;
