# Skill: spotify_play_anything

## Description
Play any artist, song, album, or playlist on Spotify using /usr/bin/spotify CLI

## Procedure
1. AUTHENTICATION (run once):
   - spotify login
   - Open browser, complete OAuth flow
   - Returns to terminal when authenticated

2. SEARCH FOR ANYTHING:
   - Artist: spotify search "Taylor Swift"
   - Song: spotify search "Blinding Lights"
   - Album: spotify search "Midnights"
   - Playlist: spotify search "Chill Vibes"

3. PLAY SEARCHED RESULTS:
   - spotify play "Artist Name" (plays artist)
   - spotify play "Song Title" (plays specific track)
   - spotify play "Playlist Name" (plays playlist)

4. PLAYBACK CONTROLS:
   - spotify pause / spotify play
   - spotify next / spotify previous
   - spotify stop
   - spotify volume [0-100]
   - spotify shuffle on/off
   - spotify repeat on/off

5. ADVANCED SEARCH:
   - spotify search "Taylor Swift:Blinding Lights" (specific track)
   - spotify search "Taylor Swift:Midnights:Track1" (album track)

6. PLAYLIST MANAGEMENT:
   - spotify playlist create "My Playlist"
   - spotify playlist add "Song1" "My Playlist"
   - spotify playlist add "Song2" "My Playlist"
   - spotify playlist remove "Song1" "My Playlist"

7. INFO COMMANDS:
   - spotify info "Artist Name"
   - spotify info "Song Title"
   - spotify info "Album Name"

8. TIPS:
   - Use quotes for search terms with spaces
   - Keep Spotify Desktop app running
   - CLI requires active session
   - Search returns IDs, use for precise playback
