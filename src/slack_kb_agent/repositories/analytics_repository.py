"""Analytics repository for tracking usage patterns and generating insights."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlalchemy import select, func, and_, or_, desc, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from .base_repository import BaseRepository
from ..database.models import (
    AnalyticsEventModel, 
    SearchResultModel, 
    UserProfileModel,
    KnowledgeBaseStatsModel
)
from ..database.connection import get_db_session
from ..exceptions import RepositoryError


logger = logging.getLogger(__name__)


class AnalyticsRepository(BaseRepository[AnalyticsEventModel]):
    """Repository for analytics events and usage tracking."""
    
    def __init__(self):
        super().__init__(AnalyticsEventModel)
    
    async def record_event(
        self,
        event_type: str,
        user_id: str,
        query: Optional[str] = None,
        response_time_ms: Optional[float] = None,
        result_count: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnalyticsEventModel:
        """Record an analytics event."""
        try:
            event_data = {
                'event_type': event_type,
                'user_id': user_id,
                'query': query,
                'response_time_ms': response_time_ms,
                'result_count': result_count,
                'success': success,
                'error_message': error_message,
                'document_id': document_id,
                'metadata': metadata or {},
                'timestamp': datetime.utcnow()
            }
            
            return await self.create(**event_data)
            
        except Exception as e:
            logger.error(f"Failed to record analytics event: {e}")
            raise RepositoryError(f"Event recording failed: {e}")
    
    async def get_user_statistics(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get user usage statistics for the specified period."""
        try:
            async with get_db_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Basic counts
                total_queries = await session.scalar(
                    select(func.count(AnalyticsEventModel.id))
                    .where(
                        and_(
                            AnalyticsEventModel.user_id == user_id,
                            AnalyticsEventModel.event_type == 'query',
                            AnalyticsEventModel.timestamp >= cutoff_date
                        )
                    )
                )
                
                successful_queries = await session.scalar(
                    select(func.count(AnalyticsEventModel.id))
                    .where(
                        and_(
                            AnalyticsEventModel.user_id == user_id,
                            AnalyticsEventModel.event_type == 'query',
                            AnalyticsEventModel.success == True,
                            AnalyticsEventModel.timestamp >= cutoff_date
                        )
                    )
                )
                
                # Average response time
                avg_response_time = await session.scalar(
                    select(func.avg(AnalyticsEventModel.response_time_ms))
                    .where(
                        and_(
                            AnalyticsEventModel.user_id == user_id,
                            AnalyticsEventModel.event_type == 'query',
                            AnalyticsEventModel.response_time_ms.isnot(None),
                            AnalyticsEventModel.timestamp >= cutoff_date
                        )
                    )
                )
                
                # Query patterns by hour
                query_by_hour = await session.execute(
                    select(
                        func.extract('hour', AnalyticsEventModel.timestamp).label('hour'),
                        func.count(AnalyticsEventModel.id).label('count')
                    )
                    .where(
                        and_(
                            AnalyticsEventModel.user_id == user_id,
                            AnalyticsEventModel.event_type == 'query',
                            AnalyticsEventModel.timestamp >= cutoff_date
                        )
                    )
                    .group_by(func.extract('hour', AnalyticsEventModel.timestamp))
                )
                
                # Most recent activity
                last_activity = await session.scalar(
                    select(func.max(AnalyticsEventModel.timestamp))
                    .where(AnalyticsEventModel.user_id == user_id)
                )
                
                return {
                    'user_id': user_id,
                    'period_days': days,
                    'total_queries': total_queries or 0,
                    'successful_queries': successful_queries or 0,
                    'success_rate': (successful_queries / total_queries) if total_queries > 0 else 0,
                    'avg_response_time_ms': float(avg_response_time) if avg_response_time else None,
                    'query_by_hour': dict(query_by_hour.fetchall()),
                    'last_activity': last_activity,
                    'generated_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get user statistics for {user_id}: {e}")
            raise RepositoryError(f"User statistics query failed: {e}")
    
    async def get_popular_queries(
        self,
        limit: int = 20,
        days: int = 7,
        min_frequency: int = 2
    ) -> List[Dict[str, Any]]:
        """Get most popular queries in the specified period."""
        try:
            async with get_db_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Group by normalized query and count occurrences
                popular_queries = await session.execute(
                    select(
                        AnalyticsEventModel.query,
                        func.count(AnalyticsEventModel.id).label('frequency'),
                        func.avg(AnalyticsEventModel.response_time_ms).label('avg_response_time'),
                        func.sum(
                            func.case((AnalyticsEventModel.success == True, 1), else_=0)
                        ).label('successful_count'),
                        func.count(distinct(AnalyticsEventModel.user_id)).label('unique_users')
                    )
                    .where(
                        and_(
                            AnalyticsEventModel.event_type == 'query',
                            AnalyticsEventModel.query.isnot(None),
                            AnalyticsEventModel.timestamp >= cutoff_date
                        )
                    )
                    .group_by(AnalyticsEventModel.query)
                    .having(func.count(AnalyticsEventModel.id) >= min_frequency)
                    .order_by(desc(func.count(AnalyticsEventModel.id)))
                    .limit(limit)
                )
                
                results = []
                for row in popular_queries:
                    results.append({
                        'query': row.query,
                        'frequency': row.frequency,
                        'avg_response_time_ms': float(row.avg_response_time) if row.avg_response_time else None,
                        'success_rate': row.successful_count / row.frequency if row.frequency > 0 else 0,
                        'unique_users': row.unique_users
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get popular queries: {e}")
            raise RepositoryError(f"Popular queries query failed: {e}")
    
    async def get_failure_patterns(
        self,
        days: int = 7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Analyze failure patterns to identify problematic queries."""
        try:
            async with get_db_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Find queries with high failure rates
                failure_patterns = await session.execute(
                    select(
                        AnalyticsEventModel.query,
                        func.count(AnalyticsEventModel.id).label('total_attempts'),
                        func.sum(
                            func.case((AnalyticsEventModel.success == False, 1), else_=0)
                        ).label('failure_count'),
                        func.array_agg(
                            distinct(AnalyticsEventModel.error_message)
                        ).label('error_messages')
                    )
                    .where(
                        and_(
                            AnalyticsEventModel.event_type == 'query',
                            AnalyticsEventModel.query.isnot(None),
                            AnalyticsEventModel.timestamp >= cutoff_date
                        )
                    )
                    .group_by(AnalyticsEventModel.query)
                    .having(
                        and_(
                            func.count(AnalyticsEventModel.id) >= 3,  # Min 3 attempts
                            func.sum(
                                func.case((AnalyticsEventModel.success == False, 1), else_=0)
                            ) > 0  # At least one failure
                        )
                    )
                    .order_by(
                        desc(
                            func.sum(
                                func.case((AnalyticsEventModel.success == False, 1), else_=0)
                            ) / func.count(AnalyticsEventModel.id)
                        )
                    )
                    .limit(limit)
                )
                
                results = []
                for row in failure_patterns:
                    failure_rate = row.failure_count / row.total_attempts if row.total_attempts > 0 else 0
                    results.append({
                        'query': row.query,
                        'total_attempts': row.total_attempts,
                        'failure_count': row.failure_count,
                        'failure_rate': failure_rate,
                        'error_messages': [msg for msg in row.error_messages if msg is not None]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get failure patterns: {e}")
            raise RepositoryError(f"Failure patterns query failed: {e}")
    
    async def get_performance_trends(
        self,
        days: int = 30,
        granularity: str = 'daily'
    ) -> Dict[str, Any]:
        """Get performance trends over time."""
        try:
            async with get_db_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Choose date grouping based on granularity
                if granularity == 'hourly':
                    date_trunc = func.date_trunc('hour', AnalyticsEventModel.timestamp)
                elif granularity == 'daily':
                    date_trunc = func.date_trunc('day', AnalyticsEventModel.timestamp)
                else:  # weekly
                    date_trunc = func.date_trunc('week', AnalyticsEventModel.timestamp)
                
                # Performance metrics over time
                trends = await session.execute(
                    select(
                        date_trunc.label('period'),
                        func.count(AnalyticsEventModel.id).label('total_queries'),
                        func.avg(AnalyticsEventModel.response_time_ms).label('avg_response_time'),
                        func.percentile_cont(0.95).within_group(
                            AnalyticsEventModel.response_time_ms
                        ).label('p95_response_time'),
                        func.sum(
                            func.case((AnalyticsEventModel.success == True, 1), else_=0)
                        ).label('successful_queries'),
                        func.count(distinct(AnalyticsEventModel.user_id)).label('active_users')
                    )
                    .where(
                        and_(
                            AnalyticsEventModel.event_type == 'query',
                            AnalyticsEventModel.timestamp >= cutoff_date
                        )
                    )
                    .group_by(date_trunc)
                    .order_by(date_trunc)
                )
                
                trend_data = []
                for row in trends:
                    success_rate = row.successful_queries / row.total_queries if row.total_queries > 0 else 0
                    trend_data.append({
                        'period': row.period.isoformat(),
                        'total_queries': row.total_queries,
                        'avg_response_time_ms': float(row.avg_response_time) if row.avg_response_time else None,
                        'p95_response_time_ms': float(row.p95_response_time) if row.p95_response_time else None,
                        'success_rate': success_rate,
                        'active_users': row.active_users
                    })
                
                return {
                    'granularity': granularity,
                    'period_days': days,
                    'data_points': trend_data,
                    'generated_at': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance trends: {e}")
            raise RepositoryError(f"Performance trends query failed: {e}")
    
    async def cleanup_old_events(
        self,
        retention_days: int = 90
    ) -> int:
        """Clean up old analytics events based on retention policy."""
        try:
            async with get_db_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                
                # Delete old events
                result = await session.execute(
                    AnalyticsEventModel.__table__.delete().where(
                        AnalyticsEventModel.timestamp < cutoff_date
                    )
                )
                
                await session.commit()
                deleted_count = result.rowcount
                
                logger.info(f"Cleaned up {deleted_count} old analytics events")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old events: {e}")
            raise RepositoryError(f"Cleanup operation failed: {e}")
    
    async def get_usage_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for usage analytics."""
        try:
            # Get data for the last 7 days and 30 days for comparison
            current_period = await self._get_period_stats(7)
            previous_period = await self._get_period_stats(7, offset_days=7)
            monthly_stats = await self._get_period_stats(30)
            
            # Calculate trends
            query_trend = self._calculate_trend(
                current_period['total_queries'], 
                previous_period['total_queries']
            )
            
            response_time_trend = self._calculate_trend(
                current_period['avg_response_time'],
                previous_period['avg_response_time'],
                inverse=True  # Lower is better for response time
            )
            
            return {
                'current_week': current_period,
                'previous_week': previous_period,
                'current_month': monthly_stats,
                'trends': {
                    'queries': query_trend,
                    'response_time': response_time_trend,
                    'success_rate': self._calculate_trend(
                        current_period['success_rate'],
                        previous_period['success_rate']
                    )
                },
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            raise RepositoryError(f"Dashboard data query failed: {e}")
    
    async def _get_period_stats(
        self, 
        days: int, 
        offset_days: int = 0
    ) -> Dict[str, Any]:
        """Get statistics for a specific period."""
        end_date = datetime.utcnow() - timedelta(days=offset_days)
        start_date = end_date - timedelta(days=days)
        
        async with get_db_session() as session:
            # Query period statistics
            stats = await session.execute(
                select(
                    func.count(AnalyticsEventModel.id).label('total_queries'),
                    func.avg(AnalyticsEventModel.response_time_ms).label('avg_response_time'),
                    func.sum(
                        func.case((AnalyticsEventModel.success == True, 1), else_=0)
                    ).label('successful_queries'),
                    func.count(distinct(AnalyticsEventModel.user_id)).label('active_users')
                )
                .where(
                    and_(
                        AnalyticsEventModel.event_type == 'query',
                        AnalyticsEventModel.timestamp >= start_date,
                        AnalyticsEventModel.timestamp < end_date
                    )
                )
            )
            
            row = stats.first()
            total_queries = row.total_queries or 0
            successful_queries = row.successful_queries or 0
            
            return {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
                'avg_response_time': float(row.avg_response_time) if row.avg_response_time else None,
                'active_users': row.active_users or 0,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
    
    def _calculate_trend(
        self, 
        current_value: Optional[float], 
        previous_value: Optional[float],
        inverse: bool = False
    ) -> Dict[str, Any]:
        """Calculate trend percentage and direction."""
        if current_value is None or previous_value is None or previous_value == 0:
            return {'percentage': 0, 'direction': 'stable', 'status': 'neutral'}
        
        percentage = ((current_value - previous_value) / previous_value) * 100
        
        if inverse:
            percentage = -percentage
        
        if percentage > 5:
            direction = 'up'
            status = 'positive'
        elif percentage < -5:
            direction = 'down'
            status = 'negative'
        else:
            direction = 'stable'
            status = 'neutral'
        
        return {
            'percentage': round(percentage, 1),
            'direction': direction,
            'status': status
        }