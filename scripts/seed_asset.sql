-- 登记本机学习用 PostgreSQL 实例，不代表生产资产或实时健康状态
INSERT INTO public.assets_asset (
    name, asset_type, ip_address
)
VALUES (
    'ops-pilot-postgres-local', 'postgresql', '127.0.0.1'
)
RETURNING id, name, asset_type, ip_address, status, created_at;