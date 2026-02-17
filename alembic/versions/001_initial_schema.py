"""Database migrations for Alembic."""
from alembic import op
import sqlalchemy as sa


def upgrade():
    op.create_table(
        'markets',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('market_id', sa.String(255), nullable=False),
        sa.Column('event_id', sa.String(255), nullable=True),
        sa.Column('series_id', sa.String(255), nullable=True),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(50), default='open'),
        sa.Column('current_price', sa.Float(), nullable=True),
        sa.Column('volume', sa.Integer(), default=0),
        sa.Column('liquidity', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('settled_at', sa.DateTime(), nullable=True),
        sa.Column('settlement_price', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('market_id')
    )
    op.create_index('idx_market_id', 'markets', ['market_id'], unique=True)
    op.create_index('idx_event_id', 'markets', ['event_id'], unique=False)

    op.create_table(
        'orders',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('order_id', sa.String(255), nullable=False),
        sa.Column('market_id', sa.Integer(), nullable=False),
        sa.Column('external_id', sa.String(255), nullable=True),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('order_type', sa.String(20), nullable=False),
        sa.Column('price', sa.Integer(), nullable=False),
        sa.Column('count', sa.Integer(), nullable=False),
        sa.Column('filled_count', sa.Integer(), default=0),
        sa.Column('remaining_count', sa.Integer(), default=0),
        sa.Column('status', sa.String(20), default='pending'),
        sa.Column('submitted_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('filled_at', sa.DateTime(), nullable=True),
        sa.Column('cancelled_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(['market_id'], ['markets.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('order_id')
    )
    op.create_index('idx_order_market_status', 'orders', ['market_id', 'status'], unique=False)
    op.create_index('idx_order_created', 'orders', ['created_at'], unique=False)

    op.create_table(
        'positions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('market_id', sa.Integer(), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('quantity', sa.Integer(), default=0),
        sa.Column('avg_cost', sa.Integer(), default=0),
        sa.Column('realized_pnl', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(['market_id'], ['markets.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_position_market_side', 'positions', ['market_id', 'side'], unique=False)

    op.create_table(
        'trades',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('trade_id', sa.String(255), nullable=False),
        sa.Column('opportunity_id', sa.String(255), nullable=True),
        sa.Column('market_id', sa.String(255), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('entry_price', sa.Integer(), nullable=False),
        sa.Column('exit_price', sa.Integer(), nullable=True),
        sa.Column('pnl', sa.Integer(), default=0),
        sa.Column('fees', sa.Integer(), default=0),
        sa.Column('status', sa.String(20), default='open'),
        sa.Column('opened_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('closed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('trade_id')
    )
    op.create_index('idx_trade_status', 'trades', ['status'], unique=False)
    op.create_index('idx_trade_opened', 'trades', ['opened_at'], unique=False)

    op.create_table(
        'portfolio_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(), default=sa.func.now()),
        sa.Column('cash_balance', sa.Integer(), default=0),
        sa.Column('positions_value', sa.Integer(), default=0),
        sa.Column('total_value', sa.Integer(), default=0),
        sa.Column('open_positions', sa.Integer(), default=0),
        sa.Column('daily_pnl', sa.Integer(), default=0),
        sa.Column('total_pnl', sa.Integer(), default=0),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_portfolio_timestamp', 'portfolio_snapshots', ['timestamp'], unique=False)


def downgrade():
    op.drop_index('idx_portfolio_timestamp', table_name='portfolio_snapshots')
    op.drop_table('portfolio_snapshots')
    op.drop_index('idx_trade_opened', table_name='trades')
    op.drop_index('idx_trade_status', table_name='trades')
    op.drop_table('trades')
    op.drop_index('idx_position_market_side', table_name='positions')
    op.drop_table('positions')
    op.drop_index('idx_order_created', table_name='orders')
    op.drop_index('idx_order_market_status', table_name='orders')
    op.drop_table('orders')
    op.drop_index('idx_event_id', table_name='markets')
    op.drop_index('idx_market_id', table_name='markets')
    op.drop_table('markets')
