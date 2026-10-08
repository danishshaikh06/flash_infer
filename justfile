default:
    @echo 'Usage:'
    @echo '  just acp "your commit message"'

acp message:
    git add .
    @echo ""
    @echo "Files staged for commit:"
    git status --short
    @echo ""
    git commit -m "{{message}}"
    git push

# Run the forward-pass test
test:
    uv run pytest tests/test2_forward.py -s

