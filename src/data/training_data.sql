CREATE OR REPLACE TEMP VIEW viewed_games AS
SELECT
  app_id,
  a.fanduel_user_id,
  event_id,
  payload.event_entities.contexts_com_fanduel_casino_global_properties_1[0].jurisdiction AS state,
  collector_tstamp_date AS event_date,
  payload.sp_analytics_entities.contexts_com_snowplowanalytics_snowplow_client_session_1.session_id[0] AS session_id,
  payload.event.search_keyword AS search_keyword,
  exploded_col.game_id AS game_id,
  exploded_col.position AS game_position,
  exploded_col.ge_version AS game_element_version
FROM foundation_views.application_events.snowplow_clean_events a
LEFT JOIN core_views.casino.users b
  ON a.fanduel_user_id = b.fanduel_user_id
LATERAL VIEW explode(
  concat(
    coalesce(
      transform(
        payload.event_entities.contexts_com_fanduel_game_element_4,
        x -> named_struct(
          'game_id', cast(x.game_id AS string),
          'position', cast(x.position AS int),
          'ge_version', 4
        )
      ),
      array()
    ),
    coalesce(
      transform(
        payload.event_entities.contexts_com_fanduel_game_element_5,
        x -> named_struct(
          'game_id', cast(x.game_id AS string),
          'position', cast(x.position AS int),
          'ge_version', 5
        )
      ),
      array()
    )
  )
) t AS exploded_col
WHERE a.collector_tstamp >= '{full_df_start_date}'
  AND event_category = 'search'
  AND event_name = 'search_results_returned'
  AND a.fanduel_user_id IS NOT NULL
  AND payload.sp_analytics_entities.contexts_com_snowplowanalytics_snowplow_client_session_1.session_id[0] IS NOT NULL
  AND b.is_customer_account
  AND NOT b.is_test_account
GROUP BY ALL;

CREATE OR REPLACE TEMP VIEW scrolled_games AS
SELECT
  app_id,
  a.fanduel_user_id,
  event_id,
  payload.event_entities.contexts_com_fanduel_casino_global_properties_1[0].jurisdiction AS state,
  collector_tstamp_date AS event_date,
  payload.sp_analytics_entities.contexts_com_snowplowanalytics_snowplow_client_session_1.session_id[0] AS session_id,
  payload.event.search_keyword AS search_keyword,
  exploded_col.game_id AS game_id,
  exploded_col.position AS game_position,
  exploded_col.ge_version AS game_element_version
FROM foundation_views.application_events.snowplow_clean_events a
LEFT JOIN core_views.casino.users b
  ON a.fanduel_user_id = b.fanduel_user_id
LATERAL VIEW explode(
  concat(
    coalesce(
      transform(
        payload.event_entities.contexts_com_fanduel_game_element_4,
        x -> named_struct(
          'game_id', cast(x.game_id AS string),
          'position', cast(x.position AS int),
          'ge_version', 4
        )
      ),
      array()
    ),
    coalesce(
      transform(
        payload.event_entities.contexts_com_fanduel_game_element_5,
        x -> named_struct(
          'game_id', cast(x.game_id AS string),
          'position', cast(x.position AS int),
          'ge_version', 5
        )
      ),
      array()
    )
  )
) t AS exploded_col
WHERE a.collector_tstamp >= '{full_df_start_date}'
  AND event_category = 'search'
  AND event_name = 'search_results_scrolled'
  AND a.fanduel_user_id IS NOT NULL
  AND payload.sp_analytics_entities.contexts_com_snowplowanalytics_snowplow_client_session_1.session_id[0] IS NOT NULL
  AND b.is_customer_account
  AND NOT b.is_test_account
GROUP BY ALL;

CREATE OR REPLACE TEMP VIEW clicked_games AS
SELECT
  app_id,
  a.fanduel_user_id,
  event_id,
  payload.event_entities.contexts_com_fanduel_casino_global_properties_1[0].jurisdiction AS state,
  collector_tstamp_date AS event_date,
  payload.sp_analytics_entities.contexts_com_snowplowanalytics_snowplow_client_session_1.session_id[0] AS session_id,
  payload.event.search_keyword AS search_keyword,
  exploded_col.game_id AS game_id,
  exploded_col.position AS game_position,
  exploded_col.ge_version AS game_element_version
FROM foundation_views.application_events.snowplow_clean_events a
LEFT JOIN core_views.casino.users b
  ON a.fanduel_user_id = b.fanduel_user_id
LATERAL VIEW explode(
  concat(
    coalesce(
      transform(
        payload.event_entities.contexts_com_fanduel_game_element_4,
        x -> named_struct(
          'game_id', cast(x.game_id AS string),
          'position', cast(x.position AS int),
          'ge_version', 4
        )
      ),
      array()
    ),
    coalesce(
      transform(
        payload.event_entities.contexts_com_fanduel_game_element_5,
        x -> named_struct(
          'game_id', cast(x.game_id AS string),
          'position', cast(x.position AS int),
          'ge_version', 5
        )
      ),
      array()
    )
  )
) t AS exploded_col
WHERE a.collector_tstamp >= '{full_df_start_date}'
  AND event_category = 'search'
  AND event_name = 'search_results_selected'
  AND a.fanduel_user_id IS NOT NULL
  AND payload.sp_analytics_entities.contexts_com_snowplowanalytics_snowplow_client_session_1.session_id[0] IS NOT NULL
  AND b.is_customer_account
  AND NOT b.is_test_account
GROUP BY ALL;

-- search_results_returned rows plus scroll-only game_ids (same user / session / app / keyword).
CREATE OR REPLACE TEMP VIEW new_agg_events AS
SELECT
  fanduel_user_id,
  session_id,
  game_id,
  event_date,
  search_keyword,
  state,
  dayofweek(event_date) AS day_of_week,
  weekofyear(event_date) AS week_of_year,
  dayofmonth(event_date) AS day_of_month,
  month(event_date) AS month_of_year,
  app_id,
  game_position
FROM (
  SELECT
    fanduel_user_id,
    session_id,
    game_id,
    event_date,
    search_keyword,
    state,
    app_id,
    game_position
  FROM viewed_games

  UNION ALL

  SELECT
    s.fanduel_user_id,
    s.session_id,
    s.game_id,
    s.event_date,
    s.search_keyword,
    s.state,
    s.app_id,
    s.game_position
  FROM scrolled_games s
  WHERE NOT EXISTS (
    SELECT 1
    FROM viewed_games v
    WHERE v.fanduel_user_id = s.fanduel_user_id
      AND v.session_id = s.session_id
      AND v.app_id = s.app_id
      AND v.game_id = s.game_id
      AND v.search_keyword = s.search_keyword
  )
) combined
GROUP BY ALL;

CREATE OR REPLACE TEMP VIEW game_popularity AS
WITH daily_stats AS (
  SELECT
    game_id,
    contest_location_code,
    contest_date,
    count(DISTINCT fanduel_user_id) AS daily_players,
    sum(bet_amount) AS daily_bet_amount,
    sum(gross_gaming_revenue) AS daily_ggr,
    sum(net_gaming_revenue) AS daily_ngr
  FROM core_views.casino.gameplay_contests
  WHERE contest_date >= current_date() - {contest_history_lookback_days}
    AND NOT is_test_account
    AND number_of_bets > 0
  GROUP BY 1, 2, 3
),
rolling AS (
  SELECT
    game_id,
    contest_location_code AS state,
    contest_date,
    daily_players,
    daily_bet_amount,
    daily_ggr,
    daily_ngr,
    avg(daily_players) OVER w AS r30_avg_daily_players,
    avg(daily_bet_amount) OVER w AS r30_avg_daily_bet_amount,
    avg(daily_ggr) OVER w AS r30_avg_daily_ggr,
    avg(daily_ngr) OVER w AS r30_avg_daily_ngr
  FROM daily_stats
  WINDOW w AS (
    PARTITION BY game_id, contest_location_code
    ORDER BY contest_date
    ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
  )
)
SELECT
  *,
  dense_rank() OVER (
    PARTITION BY state, contest_date
    ORDER BY r30_avg_daily_players DESC NULLS LAST
  ) AS r30_popularity_rank_by_num_users,
  dense_rank() OVER (
    PARTITION BY state, contest_date
    ORDER BY r30_avg_daily_bet_amount DESC NULLS LAST
  ) AS r30_popularity_rank_by_bet_amt,
  dense_rank() OVER (
    PARTITION BY state, contest_date
    ORDER BY r30_avg_daily_ggr DESC NULLS LAST
  ) AS r30_popularity_rank_by_ggr,
  dense_rank() OVER (
    PARTITION BY state, contest_date
    ORDER BY r30_avg_daily_ngr DESC NULLS LAST
  ) AS r30_popularity_rank_by_ngr
FROM rolling;

CREATE OR REPLACE TEMP VIEW game_metadata AS
WITH t1 AS (
  SELECT
    game_uid,
    state_code,
    max(inserted_ts) AS latest_ts
  FROM foundation_views.casino.prismic_games
  WHERE game_uid IS NOT NULL
  GROUP BY ALL
)
SELECT
  a.game_uid,
  a.state_code AS state,
  greatest(date_diff(day, a.release_date, current_date()), 0) AS recency,
  a.game_studio_id,
  a.game_class,
  a.game_super_class,
  a.themes,
  a.genres
FROM foundation_views.casino.prismic_games a
INNER JOIN t1
  ON a.game_uid = t1.game_uid
  AND a.state_code = t1.state_code
  AND a.inserted_ts = t1.latest_ts
WHERE a.game_uid IS NOT NULL
GROUP BY ALL;

CREATE OR REPLACE TEMP VIEW user_game_feats AS
WITH daily AS (
  SELECT
    fanduel_user_id,
    game_id,
    contest_location_code AS state,
    contest_date,
    sum(bet_amount) AS bet_amount,
    sum(number_of_bets) AS number_of_bets
  FROM core_views.casino.gameplay_contests
  WHERE contest_date >= dateadd(DAY, -{contest_history_lookback_days}, current_date())
    AND NOT is_test_account
    AND number_of_bets > 0
  GROUP BY ALL
)
SELECT
  fanduel_user_id,
  game_id,
  state,
  contest_date,
  bet_amount,
  number_of_bets,
  coalesce(
    sum(bet_amount) OVER (
      PARTITION BY fanduel_user_id, game_id, state
      ORDER BY contest_date
      RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R1_bet_amount_user_game,
  coalesce(
    sum(number_of_bets) OVER (
      PARTITION BY fanduel_user_id, game_id, state
      ORDER BY contest_date
      RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R1_num_bets_user_game,
  coalesce(
    sum(bet_amount) OVER (
      PARTITION BY fanduel_user_id, game_id, state
      ORDER BY contest_date
      RANGE BETWEEN INTERVAL 3 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R3_bet_amount_user_game,
  coalesce(
    sum(number_of_bets) OVER (
      PARTITION BY fanduel_user_id, game_id, state
      ORDER BY contest_date
      RANGE BETWEEN INTERVAL 3 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R3_num_bets_user_game,
  coalesce(
    sum(bet_amount) OVER (
      PARTITION BY fanduel_user_id, game_id, state
      ORDER BY contest_date
      RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R7_bet_amount_user_game,
  coalesce(
    sum(number_of_bets) OVER (
      PARTITION BY fanduel_user_id, game_id, state
      ORDER BY contest_date
      RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R7_num_bets_user_game,
  coalesce(
    sum(bet_amount) OVER (
      PARTITION BY fanduel_user_id, game_id, state
      ORDER BY contest_date
      RANGE BETWEEN INTERVAL 14 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R14_bet_amount_user_game,
  coalesce(
    sum(number_of_bets) OVER (
      PARTITION BY fanduel_user_id, game_id, state
      ORDER BY contest_date
      RANGE BETWEEN INTERVAL 14 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R14_num_bets_user_game,
  coalesce(
    sum(bet_amount) OVER (
      PARTITION BY fanduel_user_id, game_id, state
      ORDER BY contest_date
      RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R30_bet_amount_user_game,
  coalesce(
    sum(number_of_bets) OVER (
      PARTITION BY fanduel_user_id, game_id, state
      ORDER BY contest_date
      RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R30_num_bets_user_game
FROM daily;

CREATE OR REPLACE TEMP VIEW user_feats AS
WITH first_active AS (
  SELECT
    fanduel_user_id,
    contest_location_code AS state,
    min(contest_date) AS first_date_active
  FROM core_views.casino.gameplay_contests
  WHERE NOT is_test_account
    AND number_of_bets > 0
  GROUP BY ALL
),
daily AS (
  SELECT
    fanduel_user_id,
    contest_location_code AS state,
    contest_date,
    sum(bet_amount) AS total_bet_amount,
    sum(number_of_bets) AS total_number_of_bets
  FROM core_views.casino.gameplay_contests
  WHERE contest_date >= dateadd(DAY, -{contest_history_lookback_days}, current_date())
    AND NOT is_test_account
    AND number_of_bets > 0
  GROUP BY ALL
)
SELECT
  d.fanduel_user_id,
  d.state,
  d.contest_date,
  d.total_bet_amount,
  d.total_number_of_bets,
  f.first_date_active,
  max(d.contest_date) OVER (
    PARTITION BY d.fanduel_user_id
    ORDER BY d.contest_date
    RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 DAY PRECEDING
  ) AS latest_date_active,
  coalesce(
    sum(d.total_bet_amount) OVER (
      PARTITION BY d.fanduel_user_id
      ORDER BY d.contest_date
      RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R1_bet_amount_user,
  coalesce(
    sum(d.total_number_of_bets) OVER (
      PARTITION BY d.fanduel_user_id
      ORDER BY d.contest_date
      RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R1_num_bets_user,
  coalesce(
    sum(d.total_bet_amount) OVER (
      PARTITION BY d.fanduel_user_id
      ORDER BY d.contest_date
      RANGE BETWEEN INTERVAL 3 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R3_bet_amount_user,
  coalesce(
    sum(d.total_number_of_bets) OVER (
      PARTITION BY d.fanduel_user_id
      ORDER BY d.contest_date
      RANGE BETWEEN INTERVAL 3 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R3_num_bets_user,
  coalesce(
    sum(d.total_bet_amount) OVER (
      PARTITION BY d.fanduel_user_id
      ORDER BY d.contest_date
      RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R7_bet_amount_user,
  coalesce(
    sum(d.total_number_of_bets) OVER (
      PARTITION BY d.fanduel_user_id
      ORDER BY d.contest_date
      RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R7_num_bets_user,
  coalesce(
    sum(d.total_bet_amount) OVER (
      PARTITION BY d.fanduel_user_id
      ORDER BY d.contest_date
      RANGE BETWEEN INTERVAL 14 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R14_bet_amount_user,
  coalesce(
    sum(d.total_number_of_bets) OVER (
      PARTITION BY d.fanduel_user_id
      ORDER BY d.contest_date
      RANGE BETWEEN INTERVAL 14 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R14_num_bets_user,
  coalesce(
    sum(d.total_bet_amount) OVER (
      PARTITION BY d.fanduel_user_id
      ORDER BY d.contest_date
      RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R30_bet_amount_user,
  coalesce(
    sum(d.total_number_of_bets) OVER (
      PARTITION BY d.fanduel_user_id
      ORDER BY d.contest_date
      RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
    ),
    0
  ) AS R30_num_bets_user
FROM daily d
JOIN first_active f
  ON d.fanduel_user_id = f.fanduel_user_id
  AND d.state = f.state;

CREATE OR REPLACE TEMP VIEW base_features AS
SELECT
  a.fanduel_user_id,
  a.session_id,
  a.game_id,
  a.event_date,
  a.day_of_week,
  a.week_of_year,
  a.day_of_month,
  a.month_of_year,
  a.app_id,
  a.game_position,
  a.search_keyword,
  b.base_score AS base_similarity_score,
  b.final_score AS final_similarity_score,
  ug.R1_bet_amount_user_game,
  ug.R1_num_bets_user_game,
  ug.R3_bet_amount_user_game,
  ug.R3_num_bets_user_game,
  ug.R7_bet_amount_user_game,
  ug.R7_num_bets_user_game,
  ug.R14_bet_amount_user_game,
  ug.R14_num_bets_user_game,
  ug.R30_bet_amount_user_game,
  ug.R30_num_bets_user_game,
  datediff(DAY, u.first_date_active, current_date()) AS days_since_first_gameplay,
  datediff(DAY, u.latest_date_active, current_date()) AS days_since_last_gameplay,
  CASE WHEN year(u.first_date_active) = year(current_date()) THEN TRUE ELSE FALSE END AS first_play_this_year,
  u.R1_bet_amount_user,
  u.R1_num_bets_user,
  u.R3_bet_amount_user,
  u.R3_num_bets_user,
  u.R7_bet_amount_user,
  u.R7_num_bets_user,
  u.R14_bet_amount_user,
  u.R14_num_bets_user,
  u.R30_bet_amount_user,
  u.R30_num_bets_user,
  g.game_studio_id,
  g.game_class,
  g.game_super_class,
  g.recency,
  p.r30_popularity_rank_by_num_users,
  p.r30_popularity_rank_by_bet_amt,
  p.r30_popularity_rank_by_ggr,
  p.r30_popularity_rank_by_ngr,
  CASE WHEN click.fanduel_user_id IS NOT NULL THEN 1 ELSE 0 END AS label
FROM new_agg_events a
LEFT JOIN models.unified_search.casino_games_search_score_capture_prd_deduped b
  ON a.fanduel_user_id = b.fanduel_user_id
  AND a.event_date = b.event_date
  AND lower(trim(a.search_keyword)) = lower(trim(b.query))
  AND a.game_id = b.game_uid
LEFT JOIN user_game_feats ug
  ON a.game_id = ug.game_id
  AND upper(a.state) = ug.state
  AND a.fanduel_user_id = ug.fanduel_user_id
  AND a.event_date = ug.contest_date
LEFT JOIN user_feats u
  ON upper(a.state) = u.state
  AND a.fanduel_user_id = u.fanduel_user_id
  AND a.event_date = u.contest_date
LEFT JOIN game_metadata g
  ON a.game_id = g.game_uid
  AND upper(a.state) = g.state
LEFT JOIN game_popularity p
  ON a.game_id = p.game_id
  AND upper(a.state) = p.state
  AND a.event_date = p.contest_date
LEFT JOIN clicked_games click
  ON a.fanduel_user_id = click.fanduel_user_id
  AND a.session_id = click.session_id
  AND a.app_id = click.app_id
  AND a.game_id = click.game_id
  AND a.search_keyword = click.search_keyword
WHERE a.event_date >= dateadd(DAY, -{training_lookback_window}, current_date())
GROUP BY ALL;
