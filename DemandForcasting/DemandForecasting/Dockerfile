FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["DemandForcasting/DemandForcasting.csproj", "DemandForcasting/"]
RUN dotnet restore "DemandForcasting/DemandForcasting.csproj"
COPY . .
WORKDIR "/src/DemandForcasting"
RUN dotnet build "DemandForcasting.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "DemandForcasting.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "DemandForcasting.dll"]
