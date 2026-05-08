"""initial schema

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-05-03
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "suppliers",
        sa.Column("supplier_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("product_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("meta_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("name", name="uq_suppliers_name"),
    )

    op.create_table(
        "categories",
        sa.Column("category_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("category_name_ru", sa.String(length=255), nullable=False),
        sa.Column("category_description_ru", sa.Text(), nullable=False),
        sa.Column("normalized_text_ru", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("category_name_ru", name="uq_categories_category_name_ru"),
    )

    op.create_table(
        "products",
        sa.Column("product_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("supplier_id", sa.Integer(), sa.ForeignKey("suppliers.supplier_id"), nullable=False),
        sa.Column("original_description", sa.Text(), nullable=False),
        sa.Column("normalized_description", sa.Text(), nullable=False),
        sa.Column("source_file", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_products_supplier_id", "products", ["supplier_id"])

    op.create_table(
        "product_category_match",
        sa.Column("match_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("product_id", sa.Integer(), sa.ForeignKey("products.product_id"), nullable=False),
        sa.Column("supplier_id", sa.Integer(), sa.ForeignKey("suppliers.supplier_id"), nullable=False),
        sa.Column("category_id", sa.Integer(), sa.ForeignKey("categories.category_id"), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("product_id", "category_id", "rank", name="uq_product_category_rank"),
    )
    op.create_index("ix_product_category_match_product_id", "product_category_match", ["product_id"])
    op.create_index("ix_product_category_match_supplier_id", "product_category_match", ["supplier_id"])
    op.create_index("ix_product_category_match_category_id", "product_category_match", ["category_id"])

    op.create_table(
        "supplier_category_mapping",
        sa.Column("mapping_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("supplier_id", sa.Integer(), sa.ForeignKey("suppliers.supplier_id"), nullable=False),
        sa.Column("category_id", sa.Integer(), sa.ForeignKey("categories.category_id"), nullable=False),
        sa.Column("product_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("avg_similarity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("supplier_id", "category_id", name="uq_supplier_category"),
    )
    op.create_index("ix_supplier_category_mapping_supplier_id", "supplier_category_mapping", ["supplier_id"])
    op.create_index("ix_supplier_category_mapping_category_id", "supplier_category_mapping", ["category_id"])

    op.create_table(
        "jobs",
        sa.Column("job_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("supplier_id", sa.Integer(), sa.ForeignKey("suppliers.supplier_id"), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("jobs")

    op.drop_index("ix_supplier_category_mapping_category_id", table_name="supplier_category_mapping")
    op.drop_index("ix_supplier_category_mapping_supplier_id", table_name="supplier_category_mapping")
    op.drop_table("supplier_category_mapping")

    op.drop_index("ix_product_category_match_category_id", table_name="product_category_match")
    op.drop_index("ix_product_category_match_supplier_id", table_name="product_category_match")
    op.drop_index("ix_product_category_match_product_id", table_name="product_category_match")
    op.drop_table("product_category_match")

    op.drop_index("ix_products_supplier_id", table_name="products")
    op.drop_table("products")

    op.drop_table("categories")
    op.drop_table("suppliers")