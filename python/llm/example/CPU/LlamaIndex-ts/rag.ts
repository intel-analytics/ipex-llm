import { 
    OpenAI,
    FunctionTool,
    OpenAIAgent,
    Settings,
    SimpleDirectoryReader,
    HuggingFaceEmbedding,
    VectorStoreIndex,
    QueryEngineTool,
    Ollama,
    ReActAgent,
    OllamaEmbedding
} from "llamaindex"
import 'dotenv/config'

async function main() {
    // Settings.embedModel = new HuggingFaceEmbedding({
    //     modelType: "/mnt/disk1/models/bge-small-en-v1.5",
    //     quantized: false,
    //   });
    Settings.embedModel = new OllamaEmbedding({ model: "nomic-embed-text" });

    Settings.llm = new Ollama({
        model: "llama3:latest",
      });

    const reader = new SimpleDirectoryReader();
const documents = await reader.loadData("./data");

const index = await VectorStoreIndex.fromDocuments(documents);


const retriever = await index.asRetriever();


const queryEngine = await index.asQueryEngine({
    retriever,
  });

  const tools = [
    new QueryEngineTool({
      queryEngine: queryEngine,
      metadata: {
        name: "paper_tool",
        description: `This tool can answer detailed questions about the LLaMa 2 model.`,
      },
    }),
  ];


const agent = new ReActAgent({ tools });

let response = await agent.chat({
  message: "who are the authors of LLAMA 2",
});

console.log(response);

}

main().catch(console.error);