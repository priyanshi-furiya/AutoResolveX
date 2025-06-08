from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity, ConversationReference
from openai import AzureOpenAI
import os
import json
import traceback
import requests
from utils import generate_direct_ticket_link  # Import our utility function


class TeamsBot(ActivityHandler):
    def __init__(self, app_id: str, aoai_client=None):
        self.app_id = app_id
        self._conversation_references = {}
        self.aoai_client = aoai_client

        #  Always set this
        self.gpt_deployment = os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT')

        # Initialize Azure OpenAI client if not provided
        if not self.aoai_client:
            try:
                self.aoai_client = AzureOpenAI(
                    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
                )
                print("Azure OpenAI client initialized successfully in bot")
            except Exception as e:
                print(f"Error initializing Azure OpenAI client: {e}")
                self.aoai_client = None

    async def on_message_activity(self, turn_context: TurnContext):
        self._add_conversation_reference(turn_context.activity)
        text = turn_context.activity.text.lower() if turn_context.activity.text else ""

        try:
            if text.startswith('/'):
                # Handle commands
                await self._handle_command(turn_context, text)
            else:
                # Process with Azure OpenAI
                await self._process_with_openai(turn_context, text)
        except Exception as e:
            print(f"Error processing message: {e}")
            traceback.print_exc()
            await turn_context.send_activity("I encountered an error processing your request. Please try again.")

    async def _handle_command(self, turn_context: TurnContext, text: str):
        """Handle bot commands."""
        base_url = "https://autoresolvex-heafakckc3aqb5d6.westus3-01.azurewebsites.net"

        if text.startswith('/similar'):
            query_text = text[len('/similar'):].strip()

            # If no query text is provided, fetch default similar tickets
            if not query_text:
                try:
                    # Fetch similar tickets from the endpoint
                    await turn_context.send_activity("Fetching similar tickets...")

                    # First, try a simple query to get top similar tickets
                    response = requests.get(f"{base_url}/search-similar-incidents",
                                            params={"limit": 5})

                    # Check if the request was successful
                    if response.status_code == 200:
                        # Format and display the tickets
                        try:
                            await self._display_tickets(turn_context, "Similar Tickets", response.json())
                        except:
                            # If it's not JSON, it might be HTML
                            await self._display_tickets(turn_context, "Similar Tickets", response.text)
                    else:
                        await turn_context.send_activity(f"Error fetching similar tickets. Status code: {response.status_code}")
                except Exception as e:
                    print(f"Error fetching similar tickets: {e}")
                    traceback.print_exc()
                    await turn_context.send_activity("I encountered an error fetching similar tickets. Please try again.")
            else:
                # Search for similar incidents based on the query text
                try:
                    await turn_context.send_activity(f"Searching for tickets similar to: '{query_text}'")

                    # Make API call to search-similar-incidents endpoint
                    response = requests.post(
                        f"{base_url}/search-similar-incidents",
                        json={"text": query_text}
                    )

                    if response.status_code == 200:
                        similar_tickets = response.json()
                        if similar_tickets and len(similar_tickets) > 0:
                            await self._display_similar_search_results(turn_context, similar_tickets)
                        else:
                            await turn_context.send_activity("No similar tickets found.")
                    else:
                        await turn_context.send_activity(f"Error searching for similar tickets. Status code: {response.status_code}")
                except Exception as e:
                    print(f"Error searching for similar tickets: {e}")
                    traceback.print_exc()
                    await turn_context.send_activity("I encountered an error searching for similar tickets. Please try again.")

        elif text.startswith('/recommend'):
            try:
                # Fetch recommended tickets from the endpoint
                await turn_context.send_activity("Fetching recommended tickets...")

                # Use recommend-resolution endpoint which returns JSON
                response = requests.post(f"{base_url}/recommend-resolution",
                                         json={"summary": "High priority tickets"})

                # Check if the request was successful
                if response.status_code == 200:
                    # Format and display the tickets
                    try:
                        await self._display_recommended_tickets(turn_context, response.json())
                    except:
                        await turn_context.send_activity("Successfully fetched recommended tickets, but encountered an error displaying them.")
                else:
                    await turn_context.send_activity(f"Error fetching recommended tickets. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error fetching recommended tickets: {e}")
                traceback.print_exc()
                await turn_context.send_activity("I encountered an error fetching recommended tickets. Please try again.")

        elif text.startswith('/help'):
            help_text = """
## 🤖 How to Use This Bot

**Natural Language Queries (Recommended):**
- "Are there any similar tickets to VPN issues?"
- "Find tickets related to network problems"
- "Show me similar incidents about email issues"
- "What tickets are like password reset problems?"
- "Can you recommend solutions for printer issues?"
- "How to fix database connection errors?"

**Commands:**
- `/similar [query]` - View similar tickets or search by query
- `/recommend` - View recommended tickets
- `/help` - Show this help message

💡 **Tip:** Just ask naturally! I can understand queries like "find similar tickets about..." or "recommend solutions for..."
            """
            await turn_context.send_activity(help_text)

    async def _display_tickets(self, turn_context: TurnContext, title: str, response_data):
        """Format and display ticket data."""
        try:
            # Try to extract ticket data from the HTML response
            # This is a simplistic approach; may need adjustment based on the actual response format
            tickets = []
            base_url = "https://autoresolvex-heafakckc3aqb5d6.westus3-01.azurewebsites.net"

            # Check if response_data is actually HTML content (string)
            if isinstance(response_data, str) and '<html' in response_data.lower():
                # This is HTML content - we'll need to extract ticket data from it
                await turn_context.send_activity(f"I can see the {title.lower()} page, but I can't display the tickets here. Please use the web interface.")
                return

            # Check if tickets data can be found in the response
            if hasattr(response_data, 'get'):
                tickets = response_data.get('tickets', [])
            else:
                # If response is not JSON or dictionary-like, handle accordingly
                await turn_context.send_activity(f"Successfully fetched {title.lower()}, but could not parse the data.")
                return

            if not tickets:
                await turn_context.send_activity(f"No {title.lower()} found.")
                return

            # Format the tickets into a message
            message = f"## {title}\n\n"
            message += "|Ticket Number|Description|\n"
            message += "|---|---|\n"

            for ticket in tickets[:5]:  # Limit to 5 tickets
                ticket_number = ticket.get('Ticket Number', 'N/A')
                summary = ticket.get('Summary', 'No summary available')
                message += f"|{ticket_number}|{summary}|\n"

            # Add ticket URLs as separate lines after the table with direct access links
            message += "\n**Ticket Links:**\n"
            for ticket in tickets[:5]:
                ticket_number = ticket.get('Ticket Number', 'N/A')
                ticket_url = self._generate_direct_ticket_link(
                    base_url, ticket_number)
                message += f"- Ticket {ticket_number}: {ticket_url}\n"

            await turn_context.send_activity(message)

        except Exception as e:
            print(f"Error displaying tickets: {e}")
            traceback.print_exc()
            await turn_context.send_activity(f"Successfully fetched {title.lower()}, but encountered an error displaying them.")

    async def _display_similar_search_results(self, turn_context: TurnContext, similar_tickets):
        """Format and display similar ticket search results."""
        try:
            base_url = "https://autoresolvex-heafakckc3aqb5d6.westus3-01.azurewebsites.net"

            # Format the tickets into a message
            message = "## Similar Tickets Found\n\n"
            message += "|Ticket Number|Description|Similarity|\n"
            message += "|---|---|---|\n"

            for ticket in similar_tickets[:5]:  # Limit to 5 tickets
                ticket_number = ticket.get('TicketNumber', 'N/A')
                summary = ticket.get('Summary', 'No summary available')
                similarity = ticket.get('similarity_score', 0) * 100
                message += f"|{ticket_number}|{summary}|{similarity:.1f}%|\n"

            # Add ticket URLs as separate lines after the table with direct access links
            message += "\n**Ticket Links:**\n"
            for ticket in similar_tickets[:5]:
                ticket_number = ticket.get('TicketNumber', 'N/A')
                ticket_url = self._generate_direct_ticket_link(
                    base_url, ticket_number)
                message += f"- Ticket {ticket_number}: {ticket_url}\n"

            await turn_context.send_activity(message)

        except Exception as e:
            print(f"Error displaying similar search results: {e}")
            traceback.print_exc()
            await turn_context.send_activity("Successfully found similar tickets, but encountered an error displaying them.")

    async def _display_recommended_tickets(self, turn_context: TurnContext, response_data):
        """Format and display recommended ticket results."""
        try:
            recommendations = response_data.get('recommendations', [])
            base_url = "https://autoresolvex-heafakckc3aqb5d6.westus3-01.azurewebsites.net"

            if not recommendations:
                await turn_context.send_activity("No recommended tickets found.")
                return

            # Format the tickets into a message
            message = "## Recommended Tickets\n\n"
            message += "|Ticket Number|Description|Similarity|\n"
            message += "|---|---|---|\n"

            for rec in recommendations[:5]:  # Limit to 5 recommendations
                ticket_info = rec.get('ticket_info', {})
                ticket_number = ticket_info.get('TicketNumber', 'N/A')
                summary = ticket_info.get('Summary', 'No summary available')
                similarity = rec.get('similarity_score', 0) * 100
                message += f"|{ticket_number}|{summary}|{similarity:.1f}%|\n"

            # Add ticket URLs as separate lines after the table with direct access links
            message += "\n**Ticket Links:**\n"
            for rec in recommendations[:5]:
                ticket_info = rec.get('ticket_info', {})
                ticket_number = ticket_info.get('TicketNumber', 'N/A')
                ticket_url = self._generate_direct_ticket_link(
                    base_url, ticket_number)
                message += f"- Ticket {ticket_number}: {ticket_url}\n"

            await turn_context.send_activity(message)

        except Exception as e:
            print(f"Error displaying recommended tickets: {e}")
            traceback.print_exc()
            await turn_context.send_activity("Successfully found recommended tickets, but encountered an error displaying them.")

    async def _process_with_openai(self, turn_context: TurnContext, text: str):
        """Process message with Azure OpenAI and handle natural language queries."""
        if not text.strip():
            await turn_context.send_activity("I'm unable to process an empty message. Please provide some text.")
            return

        # First, check if this is a natural language query for similar tickets or recommendations
        if await self._handle_natural_language_query(turn_context, text):
            return

        # Check if OpenAI client is available for general chat
        if not self.aoai_client:
            print("OpenAI client not available, falling back to similar tickets search")
            await self._fallback_to_similar_search(turn_context, text)
            return

        try:
            print(f"Processing message with OpenAI: {text}")
            messages = [
                {"role": "system", "content": "You are an AI assistant that helps with IT support tickets. If users ask about similar tickets or recommendations, direct them to use natural language queries like 'find similar tickets about VPN issues' or 'show me recommendations for network problems'."},
                {"role": "user", "content": text}
            ]

            response = self.aoai_client.chat.completions.create(
                model=self.gpt_deployment,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )

            reply = response.choices[0].message.content
            print(f"Generated response: {reply}")
            await turn_context.send_activity(reply)

        except Exception as e:
            print(f"Error in OpenAI processing: {e}")
            traceback.print_exc()
            # Fall back to similar ticket search instead of showing an error
            await self._fallback_to_similar_search(turn_context, text)

    async def _fallback_to_similar_search(self, turn_context: TurnContext, query_text: str):
        """Fall back to searching for similar tickets when OpenAI processing fails."""
        base_url = "https://autoresolvex-heafakckc3aqb5d6.westus3-01.azurewebsites.net"

        try:
            await turn_context.send_activity(f"Searching for tickets similar to: '{query_text}'")

            # Make API call to search-similar-incidents endpoint
            response = requests.post(
                f"{base_url}/search-similar-incidents",
                json={"text": query_text}
            )

            if response.status_code == 200:
                similar_tickets = response.json()
                if similar_tickets and len(similar_tickets) > 0:
                    await self._display_similar_search_results(turn_context, similar_tickets)
                else:
                    await turn_context.send_activity("No similar tickets found. Try rephrasing your query or use /help to see available commands.")
            else:
                await turn_context.send_activity("I couldn't process your query directly, and there was an error finding similar tickets. Try using specific commands like /help.")
        except Exception as e:
            print(f"Error in fallback similar search: {e}")
            traceback.print_exc()
            await turn_context.send_activity("I'm having trouble processing your request. Try using specific commands like /help to see available options.")

    async def _handle_natural_language_query(self, turn_context: TurnContext, text: str):
        """Handle natural language queries for similar tickets and recommendations."""
        text_lower = text.lower()

        # Keywords that indicate a recommendation request (check first for priority)
        recommend_keywords = [
            'can you recommend', 'could you recommend', 'please recommend',
            'show me recommendations', 'give me recommendations',
            'recommend solutions', 'recommend tickets', 'recommend',
            'suggestion', 'what should', 'how to', 'best practice',
            'resolution', 'fix', 'solve', 'troubleshoot', 'help with', 'suggest'
        ]

        # Keywords that indicate a similar tickets request
        similar_keywords = [
            'similar', 'like this', 'related', 'comparable', 'same issue',
            'same problem', 'find tickets', 'search tickets', 'tickets about',
            'issues like', 'problems like', 'incidents like', 'tickets similar'
        ]

        # Check for recommendation request first (higher priority)
        is_recommend_query = any(
            keyword in text_lower for keyword in recommend_keywords)

        # Only check for similar tickets if it's not a recommendation request
        is_similar_query = not is_recommend_query and any(
            keyword in text_lower for keyword in similar_keywords)

        if is_recommend_query:
            # Extract the main topic for recommendations
            query_topic = self._extract_topic_from_query(
                text, recommend_keywords)
            print(
                f"Recommendation query detected. Original: '{text}', Extracted topic: '{query_topic}'")
            await self._get_recommendations_natural(turn_context, query_topic, text)
            return True

        elif is_similar_query:
            # Extract the main topic from the query
            query_topic = self._extract_topic_from_query(
                text, similar_keywords)
            print(
                f"Similar query detected. Original: '{text}', Extracted topic: '{query_topic}'")
            await self._search_similar_tickets_natural(turn_context, query_topic, text)
            return True

        return False

    def _extract_topic_from_query(self, text: str, keywords: list):
        """Extract the main topic from a natural language query."""
        text_lower = text.lower()

        # Remove common question words, connecting words, and action words
        stop_words = ['are', 'there', 'any', 'to', 'for', 'about',
                      'with', 'like', 'this', 'that', 'the', 'a', 'an',
                      'solutions', 'tickets', 'some', 'me', 'you', 'can',
                      'could', 'please', 'show', 'find', 'get', 'give']

        # Try to find the topic after keywords
        for keyword in keywords:
            if keyword in text_lower:
                # Find text after the keyword
                keyword_index = text_lower.find(keyword)
                after_keyword = text[keyword_index + len(keyword):].strip()

                # Clean up the extracted topic
                words = after_keyword.split()
                filtered_words = [word for word in words if word.lower(
                ) not in stop_words and len(word) > 2]

                if filtered_words:
                    # Limit to 3 relevant words for better matching
                    return ' '.join(filtered_words[:3])

        # If no specific topic found after keywords, use the whole query
        words = text.split()
        filtered_words = [word for word in words if word.lower(
        ) not in stop_words and len(word) > 2]
        return ' '.join(filtered_words[:3]) if filtered_words else text

    async def _search_similar_tickets_natural(self, turn_context: TurnContext, topic: str, original_query: str):
        """Search for similar tickets based on natural language query."""
        base_url = "https://autoresolvex-heafakckc3aqb5d6.westus3-01.azurewebsites.net"

        try:
            if topic.strip():
                await turn_context.send_activity(f"🔍 Searching for tickets similar to: **{topic}**")
                search_query = topic
            else:
                await turn_context.send_activity(f"🔍 Searching for similar tickets based on your query...")
                search_query = original_query

            # Make API call to search-similar-incidents endpoint
            response = requests.post(
                f"{base_url}/search-similar-incidents",
                json={"text": search_query}
            )

            if response.status_code == 200:
                similar_tickets = response.json()
                if similar_tickets and len(similar_tickets) > 0:
                    await self._display_similar_search_results(turn_context, similar_tickets)
                else:
                    await turn_context.send_activity("❌ No similar tickets found. Try rephrasing your query or be more specific about the issue.")
            else:
                await turn_context.send_activity("⚠️ There was an error searching for similar tickets. Please try again or use the /similar command.")
        except Exception as e:
            print(f"Error in natural language similar search: {e}")
            traceback.print_exc()
            await turn_context.send_activity("⚠️ I'm having trouble searching for similar tickets. Try using the /similar command instead.")

    async def _get_recommendations_natural(self, turn_context: TurnContext, topic: str, original_query: str):
        """Get recommendations based on natural language query."""
        base_url = "https://autoresolvex-heafakckc3aqb5d6.westus3-01.azurewebsites.net"

        try:
            if topic.strip():
                await turn_context.send_activity(f"💡 Getting recommendations for: **{topic}**")
                # Send the topic directly, similar to how similar tickets search works
                summary = topic
            else:
                await turn_context.send_activity(f"💡 Getting recommendations based on your query...")
                summary = original_query

            # Use recommend-resolution endpoint
            response = requests.post(f"{base_url}/recommend-resolution",
                                     json={"summary": summary})

            if response.status_code == 200:
                try:
                    await self._display_recommended_tickets(turn_context, response.json())
                except Exception as display_error:
                    print(f"Error displaying recommendations: {display_error}")
                    await turn_context.send_activity("✅ Found recommendations, but encountered an error displaying them. Try using the /recommend command.")
            else:
                print(
                    f"Recommendation API error: Status {response.status_code}, Response: {response.text}")
                await turn_context.send_activity("⚠️ There was an error getting recommendations. Please try again or use the /recommend command.")
        except Exception as e:
            print(f"Error in natural language recommendations: {e}")
            traceback.print_exc()
            await turn_context.send_activity("⚠️ I'm having trouble getting recommendations. Try using the /recommend command instead.")

    def _add_conversation_reference(self, activity: Activity):
        """Store a conversation reference."""
        conversation_reference = TurnContext.get_conversation_reference(
            activity)
        self._conversation_references[conversation_reference.user.id] = conversation_reference

    def _generate_direct_ticket_link(self, base_url, ticket_number):
        """Generate a direct ticket link with a security token to bypass login."""
        return generate_direct_ticket_link(base_url, ticket_number)
