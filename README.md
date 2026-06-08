# bluellm

`Claude Code` で `Azure OpenAI` (`Microsoft Foundry`) にデプロイされた GPT モデルを使用するためのプロキシです。

## インストール

```bash
curl -fsSL https://raw.githubusercontent.com/aokumablue/bluellm/main/install.sh | bash
```

## 実行

```bash
bluellm
```

## Claude Code から使用

```bash
ANTHROPIC_BASE_URL=http://localhost:8888 ANTHROPIC_API_KEY=<BLUELLM_MASTER_KEY> claude
```
