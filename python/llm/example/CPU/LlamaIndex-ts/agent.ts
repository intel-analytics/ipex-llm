import { 
    OpenAI, 
    FunctionTool, 
    OpenAIAgent,
    Settings,
    Ollama,
    ReActAgent,
} from "llamaindex"
import 'dotenv/config'

async function main() {

    // Settings.llm = new OpenAI({
    //     apiKey: process.env.OPENAI_API_KEY,
    //     model: "gpt-4o",
    // })
    Settings.llm = new Ollama({
        model: "llama3:latest",
      });
    
    /*
    Set up logging so we can see the work in progress.
    Available events:
    llm-start
    llm-end
    agent-start
    agent-end
    llm-tool-call
    llm-tool-result
    */
    Settings.callbackManager.on("llm-tool-call", (event) => {
        console.log(event.detail.payload)
    })
    Settings.callbackManager.on("llm-tool-result", (event) => {
        console.log(event.detail.payload)
    })

    const sumNumbers = ({a, b}) => {
        return `${a + b}`;
    }

    const tools = [
        FunctionTool.from(
            sumNumbers,
            {
                name: "sumNumbers",
                description: "Use this function to sum two numbers",
                parameters: {
                    type: "object",
                    properties: {
                        a: {
                            type: "number",
                            description: "First number to sum"
                        },
                        b: {
                            type: "number",
                            description: "Second number to sum"
                        },
                    },
                    required: ["a", "b"]
                }
            }
        )
    ]

    // const agent = new OpenAIAgent({tools})
    const agent = new ReActAgent({ tools });

    let response = await agent.chat({
        message: "Sum 100 and 201",
    })

    console.log(response)

}

main().catch(console.error);