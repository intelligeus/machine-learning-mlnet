FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["StockMarketPrediction/StockMarketPrediction.csproj", "StockMarketPrediction/"]
RUN dotnet restore "StockMarketPrediction/StockMarketPrediction.csproj"
COPY . .
WORKDIR "/src/StockMarketPrediction"
RUN dotnet build "StockMarketPrediction.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "StockMarketPrediction.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "StockMarketPrediction.dll"]
