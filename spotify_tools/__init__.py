from .clean import (
    fix_artists,
    reduce_list,
    column_cleaning
)

from .analysis import (
    top_songs,
    artist_summary,
    song_variability,
    feature_trend,
    feature_correlations
)

from .recommend import recommend_similar

__version__ = "0.1.0"