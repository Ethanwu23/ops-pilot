CREATE TABLE public.assets_asset(
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    name VARCHAR(100) NOT NULL,
    asset_type VARCHAR(50) NOT NULL,
    ip_address VARCHAR(45) NOT NULL,
    status VARCHAR(20) CHECK (status IN ('unknown','online', 'offline')) DEFAULT 'unknown' NOT NULL
);
