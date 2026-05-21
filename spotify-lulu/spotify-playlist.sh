#!/bin/bash

# Spotify Playlist Manager with CLI Integration
# Usage: ./spotify-playlist.sh <command> [options]

SPOTIFY_CLI="${SPOTIFY_CLI:-spotify}"
AUTH_FILE="${SPOTIFY_CLI:-spotify} --show-auth"

# Function to check if Spotify CLI is installed
check_spotify_cli() {
    if command -v "$SPOTIFY_CLI" &> /dev/null; then
        echo "✅ Spotify CLI found: $SPOTIFY_CLI"
        return 0
    else
        echo "❌ Spotify CLI not found!"
        echo ""
        echo "To install Spotify CLI:"
        echo "  macOS: brew install spotify-cli"
        echo "  Linux: sudo apt install spotify-cli"
        echo "  Windows: https://github.com/spotify/spotify-cli"
        echo ""
        echo "Or set SPOTIFY_CLI environment variable to your preferred tool"
        return 1
    fi
}

# Function to authenticate with Spotify
auth_spotify() {
    echo "🔐 Authenticating with Spotify..."
    $SPOTIFY_CLI --show-auth
    echo ""
    echo "Please follow the authentication instructions in your terminal."
    echo "After authentication, the script will be able to manage your playlists."
}

# Function to create playlist
create_playlist() {
    local name="$1"
    local desc="$2"
    local privacy="$3"
    
    if [ -z "$name" ]; then
        echo "Creating new playlist..."
        read -p "Playlist name: " name
    fi
    
    if [ -z "$desc" ]; then
        read -p "Description (optional): " desc
    fi
    
    if [ -z "$privacy" ]; then
        read -p "Privacy (public/private): " privacy
    fi
    
    if [ -z "$name" ]; then
        echo "Error: Playlist name is required!"
        return 1
    fi
    
    echo "Creating playlist: $name"
    echo "Description: $desc"
    echo "Privacy: $privacy"
    echo ""
    echo "Running: $SPOTIFY_CLI playlist create \"$name\" --description \"$desc\" --privacy \"$privacy\""
    
    if command -v "$SPOTIFY_CLI" &> /dev/null; then
        $SPOTIFY_CLI playlist create "$name" --description "$desc" --privacy "$privacy"
    fi
}

# Function to delete playlist
delete_playlist() {
    local name="$1"
    
    if [ -z "$name" ]; then
        echo "Deleting playlist..."
        read -p "Playlist name: " name
    fi
    
    if [ -z "$name" ]; then
        echo "Error: Playlist name is required!"
        return 1
    fi
    
    echo "Deleting playlist: $name"
    echo "Running: $SPOTIFY_CLI playlist delete \"$name\""
    
    if command -v "$SPOTIFY_CLI" &> /dev/null; then
        $SPOTIFY_CLI playlist delete "$name"
    fi
}

# Function to list playlists
list_playlists() {
    echo "📋 Your Playlists:"
    echo ""
    
    if command -v "$SPOTIFY_CLI" &> /dev/null; then
        $SPOTIFY_CLI playlist list
    else
        echo "Install Spotify CLI to see your playlists"
    fi
}

# Function to add song to playlist
add_song() {
    local playlist="$1"
    local song="$2"
    
    if [ -z "$playlist" ] || [ -z "$song" ]; then
        echo "Adding song to playlist..."
        read -p "Playlist name: " playlist
        read -p "Song name (or search query): " song
    fi
    
    if [ -z "$playlist" ] || [ -z "$song" ]; then
        echo "Error: Both playlist and song are required!"
        return 1
    fi
    
    echo "Adding \"$song\" to playlist \"$playlist\""
    echo "Running: $SPOTIFY_CLI playlist add \"$playlist\" \"$song\""
    
    if command -v "$SPOTIFY_CLI" &> /dev/null; then
        $SPOTIFY_CLI playlist add "$playlist" "$song"
    fi
}

# Function to remove song from playlist
remove_song() {
    local playlist="$1"
    local song="$2"
    
    if [ -z "$playlist" ] || [ -z "$song" ]; then
        echo "Removing song from playlist..."
        read -p "Playlist name: " playlist
        read -p "Song name: " song
    fi
    
    if [ -z "$playlist" ] || [ -z "$song" ]; then
        echo "Error: Both playlist and song are required!"
        return 1
    fi
    
    echo "Removing \"$song\" from playlist \"$playlist\""
    echo "Running: $SPOTIFY_CLI playlist remove \"$playlist\" \"$song\""
    
    if command -v "$SPOTIFY_CLI" &> /dev/null; then
        $SPOTIFY_CLI playlist remove "$playlist" "$song"
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "=========================================="
    echo "   🎵 Spotify Playlist Manager"
    echo "=========================================="
    echo ""
    echo "Available commands:"
    echo "  create <name>              - Create a new playlist"
    echo "  delete <name>              - Delete a playlist"
    echo "  list                       - List all playlists"
    echo "  add-song <playlist> <song> - Add song to playlist"
    echo "  remove-song <playlist> <song> - Remove song from playlist"
    echo "  auth                       - Authenticate with Spotify"
    echo "  help                       - Show this help"
    echo "  quit                       - Exit the program"
    echo ""
    echo "=========================================="
}

# Main loop
main() {
    # Check if Spotify CLI is available
    check_spotify_cli
    
    # Show menu
    show_menu
    
    while true; do
        echo ""
        read -p "Enter command: " command
        
        case "$command" in
            create)
                create_playlist
                ;;
            delete)
                delete_playlist
                ;;
            list)
                list_playlists
                ;;
            add-song)
                add_song
                ;;
            remove-song)
                remove_song
                ;;
            auth)
                auth_spotify
                ;;
            help)
                show_menu
                ;;
            quit|exit|q)
                echo "Goodbye! 👋"
                break
                ;;
            *)
                echo "Unknown command: $command"
                echo "Type 'help' for available commands"
                ;;
        esac
    done
}

# Run main function
main
