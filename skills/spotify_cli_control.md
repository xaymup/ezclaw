# Skill: spotify_cli_control

## Description
Controls Spotify playback using go-spotify-cli tool (play, pause, skip, volume, device)

## Procedure
1. Use go-spotify-cli to control Spotify playback
2. Commands available:
   - play: Starts playback on current device
   - pause: Pauses playback on current device
   - next: Skips to next track
   - previous: Returns to previous track
   - volume -v=<0-100>: Adjusts volume (e.g., volume -v=80)
   - device: Activates specific device (e.g., laptop, tablet, phone)
   - search: Search for tracks and episodes
   - saved: List and play saved tracks
3. First-time use requires Client ID and Client Secret from Spotify Developer Dashboard
4. Tokens are stored in ~/.go-spotify-cli folder
5. Use "go-spotify-cli <command>" syntax for all operations
