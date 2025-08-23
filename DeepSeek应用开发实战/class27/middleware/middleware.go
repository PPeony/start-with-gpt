// 中间件：记录请求日志
func LoggerMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        path := c.Request.URL.Path
        method := c.Request.Method

        c.Next()

        latency := time.Since(start)
        statusCode := c.Writer.Status()
        fmt.Printf("[%s] %s %d %v\n", method, path, statusCode, latency)
    }
}

// 中间件：跨域请求处理
func CorsMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
        c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

        if c.Request.Method == "OPTIONS" {
            c.AbortWithStatus(204)
            return
        }

        c.Next()
    }
}

// 中间件：请求超时控制
func TimeoutMiddleware(timeout time.Duration) gin.HandlerFunc {
    return func(c *gin.Context) {
        ctx, cancel := context.WithTimeout(c.Request.Context(), timeout)
        defer cancel()

        c.Request = c.Request.WithContext(ctx)

        done := make(chan bool, 1)
        go func() {
            c.Next()
            done <- true
        }()

        select {
        case <-done:
            return
        case <-ctx.Done():
            c.AbortWithStatusJSON(http.StatusRequestTimeout, gin.H{
                "error": "Request timeout",
            })
        }
    }
}