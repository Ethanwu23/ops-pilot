-- 授权学习用户 user-a 读取资产 1，不代表生产权限配置
INSERT INTO public.asset_access_grants (
    user_id,
    asset_id
)
VALUES (
    'user-a',
    1
)
RETURNING user_id, asset_id, created_at;